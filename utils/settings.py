"""CLI arguments, model construction and vocal-only configuration checks."""
from __future__ import annotations

import argparse
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
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--bigshifts', type=int, default=1)
    parser.add_argument('--extract_instrumental', action='store_true')
    parser.add_argument('--disable_detailed_pbar', action='store_true')
    parser.add_argument('--flac_file', action='store_true')
    parser.add_argument('--pcm_type', choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT')
    parser.add_argument('--draw_spectro', type=float, default=0)
    parser.add_argument('--filename_template', type=str, default='{file_name}/{instr}')
    args = parser.parse_args([] if dict_args is not None else None)
    if dict_args is not None:
        extra = set(dict_args) - set(vars(args))
        if extra:
            raise ValueError(f'Unsupported inference options: {sorted(extra)}')
        vars(args).update(dict_args)
    if args.model_type not in VOCAL_MODEL_TYPES:
        raise ValueError(f'Unsupported vocal model architecture: {args.model_type}')
    for field in ('config_path', 'start_check_point', 'input_folder', 'store_dir'):
        if not getattr(args, field):
            if dict_args is None:
                parser.error(f'--{field} is required')
            raise ValueError(f'Missing inference option: {field}')
    if args.bigshifts < 1:
        raise ValueError('--bigshifts must be at least 1')
    codec = 'flac' if args.flac_file else 'wav'
    if args.pcm_type not in sf.available_subtypes(codec):
        args.pcm_type = sf.default_subtype(codec)
    return args


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, encoding='utf-8') as f:
        # Trusted YAML presets may contain !!python/tuple used by upstream.
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, ConfigDict]:
    config = load_config(config_path)
    validate_vocal_config(config)
    model_type = config.training.get('model_type', model_type)
    if model_type not in VOCAL_MODEL_TYPES:
        raise ValueError(f'Unsupported vocal model architecture: {model_type}')
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
        model = BSRoformer(**dict(config.model))
    elif model_type == "mel_band_roformer":
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    else:
        raise ValueError(f"Unsupported vocal model architecture: {model_type}")
    return model, config
