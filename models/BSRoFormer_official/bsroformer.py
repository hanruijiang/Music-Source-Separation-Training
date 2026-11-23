"""BSRoFormer — Band-Split RoPE Transformer for music source separation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from rotary_embedding_torch import RotaryEmbedding

from .roformer import RMSNorm, RoFormerBlock


class BandProjection(nn.Module):
    """Per-band norm + linear projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.ff = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.ff(self.norm(x))


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


class BandSplit(nn.Module):
    """Split spectrogram into equal-width subbands and project each to a feature vector.

    Each subspec [lo, hi] is divided into n_band equal bands. Band slices are
    precomputed in the channel-interleaved f*c axis for a single-pass forward.

    Input:  (b, c, t, f)
    Output: (b, t, k, d)   k = total bands across all subspecs
    """
    def __init__(
        self, num_feature, channel_num,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec= [24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.channel_num = channel_num
        self.band_transforms = nn.ModuleList()
        self.band_slices = []  # (lo_ch, hi_ch) in the f*c axis, one per band

        for (lo, hi), n_band in zip(subspec_idxs, n_band_per_subspec):
            n_bins = hi - lo + 1
            assert n_bins % n_band == 0
            bins_per_band = n_bins // n_band
            in_dim = bins_per_band * channel_num
            for i in range(n_band):
                lo_ch = (lo + i * bins_per_band) * channel_num
                self.band_slices.append((lo_ch, lo_ch + in_dim))
                self.band_transforms.append(BandProjection(in_dim, num_feature))

    def forward(self, x):
        # x: (b, c, t, f) -> (b, t, f*c) with f as outer index
        b, c, t, f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, t, f * c)
        outs = [proj(x[:, :, lo:hi]) for (lo, hi), proj in zip(self.band_slices, self.band_transforms)]
        return torch.stack(outs).permute(1, 2, 0, 3)  # (k, b, t, d) -> (b, t, k, d)


class MaskEstimation(nn.Module):
    """Estimate a complex mask per subband; scatter into full frequency grid.

    Mirrors BandSplit: same subspec/band structure, same absolute freq_slices.
    Each band's estimate is placed into its slice; the flat mask is reshaped to (b, c, t, f).

    Input:  (b, t, k, d)
    Output: (b, c, t, f)
    """
    def __init__(
        self, num_feature, channel_num,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec=[24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.channel_num = channel_num
        self.mask_estimators = nn.ModuleList()
        self.freq_slices = []  # (lo_ch, hi_ch) in the f*c axis, one per band

        for (lo, hi), n_band in zip(subspec_idxs, n_band_per_subspec):
            n_bins = hi - lo + 1
            assert n_bins % n_band == 0
            bins_per_band = n_bins // n_band
            out_dim = bins_per_band * channel_num
            for i in range(n_band):
                lo_ch = (lo + i * bins_per_band) * channel_num
                self.freq_slices.append((lo_ch, lo_ch + out_dim))
                self.mask_estimators.append(MaskEstimator(num_feature, out_dim))

        self.num_freq_ch = self.freq_slices[-1][1]

    def forward(self, x):
        # x: (b, t, k, d)  ->  mask: (b, c, t, f)
        b, t, _, _ = x.shape
        mask = torch.zeros(b, t, self.num_freq_ch, dtype=x.dtype, device=x.device) # (b, t, f*c)
        for band_idx, ((lo, hi), estimator) in enumerate(zip(self.freq_slices, self.mask_estimators)):
            mask[:, :, lo:hi] += estimator(x[:, :, band_idx])
        # (b, t, f*c) -> (b, c, t, f)
        f = self.num_freq_ch // self.channel_num
        return mask.view(b, t, f, self.channel_num).permute(0, 3, 1, 2).contiguous()


class BSRoFormer(nn.Module):
    """Band-Split RoPE Transformer for music source separation."""
    def __init__(
        self, *, input_channels, output_channels,
        depth, num_feature, window_size, hop_size,
        subspec_idxs=[[0, 47], [48, 95], [96, 191], [192, 383], [384, 767], [768, 1023]],
        n_band_per_subspec=[24, 12, 8, 8, 8, 2],
    ):
        super().__init__()
        self.output_channels = output_channels

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            power=None, window_fn=torch.hann_window, center=True, pad_mode="reflect")
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=window_size, win_length=window_size, hop_length=hop_size,
            window_fn=torch.hann_window, center=True, pad_mode="reflect")

        stft_channels = input_channels * 2  # real + imag
        dim_head = num_feature // 8

        self.multi_band_transform = BandSplit(
            num_feature, stft_channels, subspec_idxs, n_band_per_subspec)
        self.rotary_emb_t = RotaryEmbedding(dim=dim_head)
        self.rotary_emb_k = RotaryEmbedding(dim=dim_head)
        self.transformer_stack = nn.ModuleList([
            RoFormerBlock(num_feature, 8, dim_head, num_feature * 4, 0)
            for _ in range(depth)
        ])
        self.mask_estimation = MaskEstimation(
            num_feature, output_channels * 2, subspec_idxs, n_band_per_subspec)

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
        band_features = self.multi_band_transform(band_input)   # (b, t, k, d)

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
