# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import os
import glob
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import bigshifts_wrapper
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint, compile_model




def run_folder(
    model: "torch.nn.Module",
    args: "argparse.Namespace",
    config: dict,
    device: "torch.device",
    verbose: bool = False
) -> list[str]:
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()
    written_files = []
    claimed_paths = set()
    failed_files = 0

    input_root = Path(args.input_folder).resolve()
    output_root = Path(args.store_dir).resolve()
    if not input_root.is_dir():
        raise ValueError(f'Input folder does not exist: {args.input_folder}')
    if input_root == output_root:
        raise ValueError('Input and output folders must be different')

    # Recursively collect files, excluding previous results inside the input tree.
    mixture_paths = sorted(
        glob.glob(os.path.join(args.input_folder, "**/*.*"), recursive=True)
    )
    mixture_paths = [p for p in mixture_paths if os.path.isfile(p)
                     and not (output_root.is_relative_to(input_root)
                              and Path(p).resolve().is_relative_to(output_root))]
    if not mixture_paths:
        raise ValueError(f'No input files found: {args.input_folder}')

    input_paths = {Path(path).resolve() for path in mixture_paths}
    sample_rate: int = getattr(config.audio, "sample_rate", 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments: list[str] = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    # Determine whether to use detailed progress bar
    detailed_pbar = not args.disable_detailed_pbar

    for path in mixture_paths:
        # Get relative path from input folder
        relative_path: str = os.path.relpath(path, args.input_folder)
        # Extract directory and file name
        dir_name: str = os.path.dirname(relative_path) or '.'
        file_name: str = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f"Cannot read track: {format(path)}")
            print(f"Error message: {str(e)}")
            failed_files += 1
            continue

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if config.audio.get('num_channels', 2) == 2:
                print('Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

        expected_channels = config.audio.get('num_channels', 2)
        if mix.ndim != 2 or mix.shape[0] != expected_channels or mix.shape[-1] == 0:
            print(f'Skip unsupported or empty audio: {path}')
            failed_files += 1
            continue

        mix_orig = mix.copy()

        # Normalize input audio if enabled
        normalize = config.inference.get('normalize', False)
        if normalize:
            mix, norm_params = normalize_audio(mix)

        # Perform source separation
        waveforms_orig = bigshifts_wrapper(
            config,
            model,
            mix,
            device,
            pbar=detailed_pbar,
            bigshifts=args.bigshifts
        )

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                bigshifts=args.bigshifts,
                pbar=detailed_pbar
            )

        # Extract instrumental track if requested
        if args.extract_instrumental:
            instr = "vocals"
            vocals_estimate = waveforms_orig[instr]
            if normalize:
                vocals_estimate = denormalize_audio(vocals_estimate, norm_params)
            waveforms_orig["instrumental"] = mix_orig - vocals_estimate
            if "instrumental" not in instruments:
                instruments.append("instrumental")

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if normalize and not (instr == "instrumental" and args.extract_instrumental):
                estimates = denormalize_audio(estimates, norm_params)

            codec = "flac" if args.flac_file else "wav"

            subtype = args.pcm_type

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                source_name=os.path.basename(path),
                dir_name=dir_name,
                model_type=args.model_type,
                model=os.path.splitext(
                    os.path.basename(args.start_check_point)
                )[0],
            )

            # Create output directory
            output_dir: str = os.path.join(args.store_dir, *dirnames)
            output_path: str = os.path.join(output_dir, f"{fname}.{codec}")
            resolved = Path(output_path).resolve()
            if not resolved.is_relative_to(output_root):
                raise ValueError(f'Output path escapes store_dir: {output_path}')
            if resolved in input_paths or resolved in claimed_paths:
                raise ValueError(f'Conflicting output path: {output_path}; choose a unique filename_template')
            claimed_paths.add(resolved)
            os.makedirs(output_dir, exist_ok=True)
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            written_files.append(output_path)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                resolved_img = Path(output_img_path).resolve()
                if (not resolved_img.is_relative_to(output_root)
                        or resolved_img in claimed_paths or resolved_img in input_paths):
                    raise ValueError(f'Conflicting spectrogram path: {output_img_path}')
                claimed_paths.add(resolved_img)
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    if not written_files:
        raise RuntimeError("No audio files were successfully processed")
    if failed_files:
        print(f"Warning: {failed_files} input file(s) could not be processed")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")
    return written_files

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        if any(i >= torch.cuda.device_count() for i in args.device_ids):
            raise ValueError('CUDA device ID out of range')
        device = f'cuda:{args.device_ids[0]}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
    load_start_checkpoint(model, checkpoint)
    del checkpoint

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if torch.device(device).type == 'cuda' and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    if args.compile:
        if isinstance(model, nn.DataParallel) or device in ("cpu", "mps"):
            print('Warning: --compile is supported only for single CUDA GPU, ignore it')
        else:
            model = compile_model(model.eval())

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    return run_folder(model, args, config, device, verbose=True)


def separate_folder(*, model_type: str, config_path: str,
                    checkpoint_path: str, input_folder: str,
                    output_folder: str, **options) -> list[str]:
    """Run vocal separation from Python and return written audio paths.

    Example: separate_folder(model_type='bs_roformer',
        config_path='configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml',
        checkpoint_path='weights.ckpt', input_folder='audio', output_folder='out')
    """
    params = dict(model_type=model_type, config_path=str(config_path),
                  start_check_point=str(checkpoint_path),
                  input_folder=str(input_folder), store_dir=str(output_folder))
    params.update(options)
    return proc_folder(params)


def main() -> None:
    proc_folder(None)


if __name__ == "__main__":
    main()
