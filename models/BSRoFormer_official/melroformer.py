"""MelRoformer — Mel-band RoPE Transformer for music source separation."""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from rotary_embedding_torch import RotaryEmbedding

from .roformer import RMSNorm, RoFormerBlock


def _mel_band_ranges(n_fft, num_bands):
    """Compute overlapping frequency bin ranges for each mel band.

    Returns list of [lo_bin, hi_bin] pairs mapping mel bands to FFT bins.
    """
    filterbank = librosa.filters.mel(sr=44100, n_fft=n_fft, n_mels=num_bands)
    band_bins = [list(np.where(row > 0)[0]) for row in filterbank]
    band_bins[0] = [0] + band_bins[0]  # include DC bin
    return [[bins[0], bins[-1]] for bins in band_bins]


class BandProjection(nn.Module):
    """Per-band norm + linear projection used in MelBandSplit."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.ff = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.ff(self.norm(x))


class MelBandSplit(nn.Module):
    """Split spectrogram into mel-scale frequency bands and project each to a feature vector."""

    def __init__(self, feature_dim, num_channels, band_ranges):
        super().__init__()
        self.num_channels = num_channels
        self.band_ranges = band_ranges  # [(lo_bin, hi_bin), ...]
        self.mel_band_transforms = nn.ModuleList([
            BandProjection((hi - lo + 1) * num_channels, feature_dim)
            for lo, hi in band_ranges
        ])

    def forward(self, x):
        # x: (b, c, t, f) -> (b, t, f*c) with f as outer index
        b, c, t, f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, t, f * c)
        ch = self.num_channels
        band_features = [
            proj(x[:, :, lo * ch : (hi + 1) * ch])
            for (lo, hi), proj in zip(self.band_ranges, self.mel_band_transforms)
        ]
        # (k, b, t, d) -> (b, t, k, d)
        return torch.stack(band_features).permute(1, 2, 0, 3)


class MaskEstimator(nn.Module):
    """Per-band mask estimator: norm -> linear -> tanh -> linear -> GLU."""

    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(feature_dim)
        self.ff_0 = nn.Linear(feature_dim, 4 * feature_dim)
        self.tanh = nn.Tanh()
        self.ff_1 = nn.Linear(4 * feature_dim, 2 * out_dim)
        self.glu = nn.GLU()

    def forward(self, x):
        return self.glu(self.ff_1(self.tanh(self.ff_0(self.norm(x)))))


class MelBandMask(nn.Module):
    """Estimate a complex mask for each mel band; scatter into full frequency grid.

    For each band k, a MaskEstimator maps the band's feature vector (b, t, d) to
    a mask chunk of size (freq_hi - freq_lo) covering its channel-interleaved
    frequency slice. Chunks are accumulated into a full-size mask, then reshaped
    from (b, t, f*c) -> (b, c, t, f).
    """

    def __init__(self, feature_dim, num_channels, band_ranges):
        super().__init__()
        self.num_channels = num_channels
        # Channel-interleaved slices: (lo_ch, hi_ch) in the flattened f*c axis.
        # lo_ch = lo_bin * c,  hi_ch = (hi_bin + 1) * c  ->  slice size = (hi_bin - lo_bin + 1) * c
        self.freq_slices = [
            (int(lo) * num_channels, (int(hi) + 1) * num_channels)
            for lo, hi in band_ranges
        ]
        self.num_freq_ch = self.freq_slices[-1][1]  # total size of the f*c axis
        self.mask_estimations = nn.ModuleList([
            MaskEstimator(feature_dim, hi - lo)
            for lo, hi in self.freq_slices
        ])

    def forward(self, x):
        # x: (b, t, k, d)  ->  mask: (b, c, t, f)
        b, t, _, _ = x.shape
        mask = torch.zeros(b, t, self.num_freq_ch, dtype=x.dtype, device=x.device)
        for band_idx, ((lo, hi), estimator) in enumerate(
            zip(self.freq_slices, self.mask_estimations)
        ):
            mask[:, :, lo:hi] += estimator(x[:, :, band_idx])
        # (b, t, f*c) -> (b, c, t, f)
        f = self.num_freq_ch // self.num_channels
        return mask.view(b, t, f, self.num_channels).permute(0, 3, 1, 2).contiguous()


class MelRoFormer(nn.Module):
    """Mel-band RoPE Transformer for music source separation."""

    def __init__(self, *, input_channels, output_channels,
                 depth, num_feature, window_size, hop_size, mel_bands):
        super().__init__()
        self.output_channels = output_channels

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            power=None, window_fn=torch.hann_window, center=True, pad_mode="reflect")
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            window_fn=torch.hann_window, center=True, pad_mode="reflect")

        band_ranges = _mel_band_ranges(window_size - 1, mel_bands)
        stft_channels = input_channels * 2  # real + imag
        dim_head = num_feature // 8

        self.multi_band_transform = MelBandSplit(num_feature, stft_channels, band_ranges)
        self.rotary_emb_t = RotaryEmbedding(dim=dim_head)
        self.rotary_emb_k = RotaryEmbedding(dim=dim_head)
        self.transformer_stack = nn.ModuleList([
            RoFormerBlock(num_feature, 8, dim_head, num_feature * 4, 0)
            for _ in range(depth)
        ])
        self.mask_estimation = MelBandMask(
            num_feature, output_channels * 2, band_ranges)

    def forward(self, mix):
        """mix: (batch, channels, samples) -> separated: (batch, channels, samples)"""
        batch_size = mix.shape[0]

        # STFT -> complex spectrogram; disable autocast (requires float32)
        with torch.autocast(device_type="cuda", enabled=False):
            complex_spec = self.stft(mix.float())
        # (b, c, f, t) -> (b, c, t, f)
        real = complex_spec.real.transpose(2, 3)
        imag = complex_spec.imag.transpose(2, 3)

        # Band split: spectrogram -> per-band feature vectors
        band_input = torch.cat((real, imag), dim=1)[..., :-1]  # drop Nyquist bin
        band_features = self.multi_band_transform(band_input)   # (b, t, bands, d)

        # Alternating time/band transformers with rotary embeddings
        b, t, k, d = batch_size, *band_features.shape[1:]
        for block in self.transformer_stack:
            # (b, t, k, d) -> (b*k, t, d): attend across time within each band
            band_features = block.transform_t(
                band_features.permute(0, 2, 1, 3).contiguous().reshape(b * k, t, d),
                self.rotary_emb_t)
            # (b*k, t, d) -> (b*t, k, d): attend across bands within each time step
            band_features = block.transform_k(
                band_features.view(b, k, t, d).permute(0, 2, 1, 3).contiguous().reshape(b * t, k, d),
                self.rotary_emb_k)
            # (b*t, k, d) -> (b, t, k, d)
            band_features = band_features.view(b, t, k, d)

        # Estimate complex mask and apply to input spectrogram
        complex_mask = F.pad(self.mask_estimation(band_features), (0, 1))  # restore Nyquist
        mask_real = complex_mask[:, :self.output_channels]
        mask_imag = complex_mask[:, self.output_channels:]

        with torch.autocast(device_type="cuda", enabled=False):
            separated_real = real.float() * mask_real.float() - imag.float() * mask_imag.float()
            separated_imag = real.float() * mask_imag.float() + imag.float() * mask_real.float()
            # (b, c, t, f) -> (b, c, f, t) for iSTFT
            audio = self.istft(torch.complex(
                separated_real.transpose(2, 3).contiguous(),
                separated_imag.transpose(2, 3).contiguous()), length=mix.shape[-1])

        return audio.to(mix.dtype)


# ── Tests ────────────────────────────────────────────────────
