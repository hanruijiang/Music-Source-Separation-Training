"""CLI arguments, model construction and vocal-only configuration checks."""
from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import soundfile as sf
import yaml
from torch import nn
from ml_collections import ConfigDict

VOCAL_MODEL_TYPES = frozenset(("mdx23c", "segm_models", "bs_roformer", "mel_band_roformer", "bs_polarformer"))


def validate_vocal_config(config) -> None:
    """Allow only vocals, optionally accompanied by other/instrumental."""
    training = getattr(config, 'training', None)
    if training is None:
        raise ValueError('Expected training.instruments in vocal checkpoint config')
    instruments = list(training.get('instruments', []))
    if not instruments or 'vocals' not in instruments:
        raise ValueError('Only Vocal models are supported (vocals is required)')
    if len(set(instruments)) != len(instruments):
        raise ValueError('Duplicate stems in vocal configuration')
    unexpected = set(instruments) - {'vocals', 'other', 'instrumental'}
    if unexpected:
        raise ValueError(f'Non-vocal stems are unsupported: {sorted(unexpected)}')
    if 'other' in instruments and 'instrumental' in instruments:
        raise ValueError('Specify only one accompaniment stem: other or instrumental')
    target = training.get('target_instrument', None)
    if target not in (None, 'vocals'):
        raise ValueError(f'Non-vocal target instrument: {target}')


def parse_args_inference(dict_args: Dict | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Vocal-only inference: CLI or Python API')
    parser.add_argument('--model_type', type=str, default='mdx23c', choices=sorted(VOCAL_MODEL_TYPES))
    parser.add_argument('--config_path', type=str, help='Path to a vocal YAML preset')
    parser.add_argument('--start_check_point', type=str, help='Full model checkpoint')
    parser.add_argument('--input_folder', type=str, help='Input audio folder')
    parser.add_argument('--store_dir', type=str, help='Output audio folder')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='CUDA devices')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--optimization', choices=('off', 'safe', 'advanced'), default='off',
                        help='Opt-in inference model optimization; advanced also tries CUDA channels-last')
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--bigshifts', type=int, default=1)
    parser.add_argument('--extract_instrumental', action='store_true')
    parser.add_argument('--disable_detailed_pbar', action='store_true')
    parser.add_argument('--flac_file', action='store_true')
    parser.add_argument('--pcm_type', choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT')
    parser.add_argument('--draw_spectro', type=float, default=0)
    parser.add_argument('--filename_template', type=str, default='{dir_name}/{file_name}/{instr}')
    args = parser.parse_args([] if dict_args is not None else None)
    if dict_args is not None:
        extra = set(dict_args) - set(vars(args))
        if extra:
            raise ValueError(f'Unsupported inference options: {sorted(extra)}')
        vars(args).update(dict_args)
    if args.optimization not in ('off', 'safe', 'advanced'):
        raise ValueError('Unsupported optimization mode')
    if args.model_type not in VOCAL_MODEL_TYPES:
        raise ValueError(f'Unsupported vocal model architecture: {args.model_type}')
    for field in ('config_path', 'start_check_point', 'input_folder', 'store_dir'):
        if not getattr(args, field):
            if dict_args is None:
                parser.error(f'--{field} is required')
            raise ValueError(f'Missing inference option: {field}')
    if type(args.bigshifts) is not int or args.bigshifts < 1:
        raise ValueError('--bigshifts must be a positive integer')
    if not isinstance(args.draw_spectro, (int, float)) or not math.isfinite(args.draw_spectro) or args.draw_spectro < 0:
        raise ValueError('--draw_spectro must be finite and nonnegative')
    if type(args.device_ids) is int:
        args.device_ids = [args.device_ids]
    if (not isinstance(args.device_ids, (list, tuple)) or not args.device_ids
            or any(type(i) is not int or i < 0 for i in args.device_ids)
            or len(set(args.device_ids)) != len(args.device_ids)):
        raise ValueError('--device_ids must contain distinct nonnegative integers')
    args.device_ids = list(args.device_ids)
    if args.pcm_type not in ('PCM_16', 'PCM_24', 'FLOAT'):
        raise ValueError('Unsupported PCM type')
    if not isinstance(args.filename_template, str) or not args.filename_template.strip():
        raise ValueError('--filename_template must be a nonempty string')
    codec = 'flac' if args.flac_file else 'wav'
    if args.pcm_type not in sf.available_subtypes(codec):
        args.pcm_type = sf.default_subtype(codec)
    return args


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, encoding='utf-8') as f:
        # Trusted YAML presets may contain !!python/tuple used by upstream.
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def validate_inference_config(config) -> None:
    """Reject invalid parameters before constructing a potentially large model."""
    chunk_size = config.inference.get('chunk_size', config.audio.get('chunk_size'))
    overlap = config.inference.get('num_overlap')
    batch_size = config.inference.get('batch_size')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if type(overlap) is not int or not 1 <= overlap <= chunk_size:
        raise ValueError('num_overlap must be an integer between 1 and chunk_size')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if type(config.audio.get('sample_rate', 44100)) is not int or config.audio.get('sample_rate', 44100) <= 0:
        raise ValueError('sample_rate must be a positive integer')
    channels = config.audio.get('num_channels', 2)
    if type(channels) is not int or channels not in (1, 2):
        raise ValueError('num_channels must be 1 or 2')
    if 'stereo' in config.model and channels != (2 if config.model.stereo else 1):
        raise ValueError('audio.num_channels does not match model.stereo')
    target = config.training.get('target_instrument')
    expected_stems = 1 if target else len(config.training.instruments)
    if 'num_stems' in config.model and config.model.num_stems != expected_stems:
        raise ValueError('model.num_stems does not match the configured output stems')


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, ConfigDict]:
    config = load_config(config_path)
    validate_vocal_config(config)
    validate_inference_config(config)
    if model_type not in VOCAL_MODEL_TYPES:
        raise ValueError(f'Unsupported vocal model architecture: {model_type}')
    if model_type in ("mdx23c", "segm_models") and config.audio.get("num_channels", 2) != 2:
        raise ValueError("MDX23C and Segm Models require stereo audio")
    if model_type == "mdx23c":
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == "segm_models":
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type in ("bs_roformer", "bs_polarformer"):
        from models.bs_roformer import BSRoformer
        if model_type == "bs_polarformer" and not config.model.get("use_pope", False):
            raise ValueError("BS PolarFormer requires model.use_pope: true")
        if model_type == "bs_roformer" and config.model.get("use_pope", False):
            raise ValueError("Select bs_polarformer for model.use_pope: true")
        model = BSRoformer(**dict(config.model))
    elif model_type == "mel_band_roformer":
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    else:
        raise ValueError(f"Unsupported vocal model architecture: {model_type}")
    return model, config
