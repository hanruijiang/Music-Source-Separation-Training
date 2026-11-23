"""Inference-only demixing, augmentation and checkpoint utilities."""
import argparse
import os
import shutil
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from tqdm.auto import tqdm


def compile_model(model: torch.nn.Module, inplace: bool = False) -> torch.nn.Module:
    """
    Use torch.compile for faster inference (fuses elementwise ops: norms, rotary, gates, casts).

    For roformer-like models (layers + band_split + mask_estimators) submodules are compiled in place:
    all transformer blocks share one compiled graph, so compilation takes seconds instead of minutes
    (whole-model compile is only ~4% faster but needs ~1.5 min of tracing on every run).
    Shapes are static: demix pads every batch to `inference.batch_size`, so there are no recompilations.
    inplace=True compiles other models in place too (state_dict keys stay the same, for checkpoint compatibility).
    Don't use it with DataParallel: replicas share the compiled function bound to the original module,
    so they would silently run the weights of the GPU 0 copy.
    """
    # triton builds its helper module with the compiler from CC env var; if it points to a missing
    # executable (e.g. "cc -mavx2" on Windows), drop it so triton uses its bundled TinyCC / MSVC
    cc = os.environ.get("CC")
    if cc and shutil.which(cc.split()[0]) is None:
        print(f"torch.compile: CC='{cc}' not found, ignore it for triton")
        os.environ.pop("CC")

    # persistent on-disk cache of compiled graphs and triton kernels (separate entries for each GPU arch),
    # so only the first run compiles from scratch. Can be overridden by setting env vars before the run.
    # Some imports already fill TORCHINDUCTOR_CACHE_DIR with torch default (temp dir), so replace the default too
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") in (None, default_cache_dir()):
        code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = code_root + '/.torch_compile_cache'
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    print(f"torch.compile cache dir: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")

    if all(hasattr(model, name) for name in ("layers", "band_split", "mask_estimators")):
        print("Use torch.compile for model submodules (first batch will be slower due to compilation)")
        for layer in model.layers:
            for block in layer:
                block.compile(dynamic=False)
        model.band_split.compile(dynamic=False)
        for mask_estimator in model.mask_estimators:
            mask_estimator.compile(dynamic=False)
        model.compiled_submodules = True
        return model

    print("Use torch.compile for model (first batch will be slow due to compilation)")
    if inplace:
        model.compile(dynamic=False)
        model.compiled_submodules = True
        return model
    return torch.compile(model, dynamic=False)


def is_compiled_model(model: torch.nn.Module) -> bool:
    model = getattr(model, "module", model)  # DDP / DataParallel wrapper
    return hasattr(model, "_orig_mod") or getattr(model, "compiled_submodules", False)


def bigshifts_wrapper(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False,
    bigshifts: int = 1
) -> Dict[str, np.ndarray]:
    """BigShifts wrapper for inference-time demixing."""

    if bigshifts <= 0:
        bigshifts = 1

    if isinstance(mix, torch.Tensor):
        mix = mix.detach().cpu().numpy()

    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []

    if pbar:
        shifts_iterator = tqdm(shifts, desc="BigShifts passes...", leave=False)
    else:
        shifts_iterator = shifts

    for shift in shifts_iterator:
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix(config, model, shifted_mix, device, model_type, pbar)

        unshifted = {
            k: np.concatenate((v[..., shift:], v[..., :shift]), axis=-1)
            for k, v in sources.items()
        }
        results.append(unshifted)

    return {k: np.mean([r[k] for r in results], axis=0) for k in results[0]}


def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    model_type: str,
    pbar: bool = False
) -> Dict[str, np.ndarray]:
    """
    Perform audio source separation with a given model.

    Supports overlapping chunk-based inference with optional progress bar display.
    Handles padding, fading, and batching to reduce artifacts during separation.

    Args:
        config (ConfigDict): Configuration object with audio and inference
            parameters (chunk size, overlap, batch size, etc.).
        model (torch.nn.Module): Source separation model for inference.
        mix (torch.Tensor): Input audio tensor of shape (channels, time).
        device (torch.device): Device on which to run inference (CPU or CUDA).
        model_type (str): Architecture identifier for output-shape handling.
        pbar (bool, optional): If True, show a progress bar during chunk
            processing. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Mapping of instrument names to estimated waveforms.
    """

    mix = torch.tensor(mix, dtype=torch.float32)

    num_instruments = len(prefer_target_instrument(config))

    # keep whole track and accumulators on GPU (avoids per chunk transfers and slow CPU accumulation)
    acc_device = _get_accumulation_device(device, (num_instruments + 1) * mix.numel() * 4 * 1.5)
    mix = mix.to(acc_device)

    # All retained architectures use overlapping chunk-based inference.
    if 'chunk_size' in config.inference:
        chunk_size = config.inference.chunk_size
    else:
        chunk_size = config.audio.chunk_size
    num_overlap = config.inference.num_overlap
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = _getWindowingArray(chunk_size, fade_size).to(acc_device)
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size
    pad_batch = is_compiled_model(model)

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=(use_amp and torch.device(device).type == 'cuda')):
        # Use no_grad because some models populate mutable lazy inference caches.
        with torch.no_grad():
            # Initialize result and counter tensors (counter is the same for all instruments and channels)
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32, device=acc_device)
            counter = torch.zeros(mix.shape[-1], dtype=torch.float32, device=acc_device)

            i = 0
            batch_data = []
            batch_locations = []
            if pbar:
                progress_bar = tqdm(
                    total=mix.shape[1], desc="Processing audio chunks", leave=False
                )
            else:
                progress_bar = None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    if pad_batch and arr.shape[0] < batch_size:
                        # keep batch shape constant for compiled model (avoid recompilation)
                        arr = nn.functional.pad(arr, (0, 0, 0, 0, 0, batch_size - arr.shape[0]))
                    x = model(arr).to(acc_device, torch.float32)
                    if x.ndim == 3 and model_type in ("bs_roformer_official", "mel_roformer_official"):
                        x = x.unsqueeze(1)

                    for j, (start, seg_len) in enumerate(batch_locations):
                        first = start == 0
                        last = start + step >= mix.shape[1]
                        window = windowing_array.clone() if fade_size and (first or last) else windowing_array
                        if fade_size and first:
                            window[:fade_size] = 1
                        if fade_size and last:
                            window[-fade_size:] = 1
                        result[..., start:start + seg_len] += x[j, ..., :seg_len] * window[..., :seg_len]
                        counter[start:start + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result.div_(counter)

            # Remove overlap padding.
            if length_init > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]

            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    return dict(zip(prefer_target_instrument(config), estimated_sources))

def apply_tta(
    config,
    model: torch.nn.Module,
    mix: torch.Tensor,
    waveforms_orig: dict[str, np.ndarray],
    device: torch.device,
    model_type: str,
    bigshifts: int = 1,
    pbar: bool = False
) -> dict[str, np.ndarray]:
    """
    Enhance source separation results using Test-Time Augmentation (TTA).

    Applies augmentations such as channel reversal and polarity inversion to
    the input mixture, reprocesses with the model, and combines the results
    with the original predictions by averaging.

    Args:
        config: Configuration object with model and inference parameters.
        model (torch.nn.Module): Trained source separation model.
        mix (torch.Tensor): Input mixture tensor of shape (channels, time).
        waveforms_orig (Dict[str, torch.Tensor]): Dictionary of separated
            sources before augmentation.
        device (torch.device): Computation device (CPU or CUDA).
        model_type (str): Model type identifier used for demixing.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of separated sources after applying TTA.
    """

    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = bigshifts_wrapper(
            config,
            model,
            augmented_mix,
            device,
            model_type=model_type,
            bigshifts=bigshifts,
            pbar=pbar
        )
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _get_accumulation_device(device: Union[torch.device, str], required_bytes: float) -> torch.device:
    """
    Choose device to store the whole track and overlap-add accumulators in demix.

    Use the CUDA device if the buffers take less than half of its available memory
    (free + cached by PyTorch allocator), otherwise fall back to CPU (e.g. for very long tracks).
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return torch.device('cpu')
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return device if required_bytes < 0.5 * available else torch.device('cpu')


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
        Return the list of target instruments based on the configuration.
        If a specific target instrument is specified in the configuration,
        it returns a list with that instrument. Otherwise, it returns the list of instruments.

        Parameters:
        ----------
        config : ConfigDict
            Configuration object containing the list of instruments or the target instrument.

        Returns:
        -------
        List[str]
            A list of target instruments.
        """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments


def strip_compile_prefix(state_dict: dict) -> dict:
    """
    Remove the `_orig_mod.` parts that torch.compile (not in place) adds to state_dict keys,
    e.g. "blocks.0._orig_mod.time_block.norm1.scale" -> "blocks.0.time_block.norm1.scale".
    Without it such checkpoints load silently without these weights in train mode and fail in valid mode.
    """
    if not any('_orig_mod.' in k for k in state_dict):
        return state_dict
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}



def load_start_checkpoint(args: argparse.Namespace, model: torch.nn.Module,
                          old_model, type_: str = 'inference') -> None:
    """Load a **full** inference checkpoint, strictly, without LoRA/fine-tuning.

    Accepts raw state dicts as well as wrapped checkpoint dicts. Only load
    checkpoints from trusted sources (torch.load may unpickle Python objects).
    """
    if type_ != 'inference':
        raise ValueError('This build supports inference only')
    state = old_model
    for key in ('state', 'state_dict', 'model_state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a model state dict')
    model.load_state_dict(strip_compile_prefix(state), strict=True)
