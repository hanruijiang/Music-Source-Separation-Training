#!/usr/bin/env python3
"""Reduce the Music-Source-Separation-Training repository to vocal inference.

Default: preview only. --apply: modify a clean Git checkout on a feature branch.
The script neither commits nor pushes to GitHub.

This combines both user requests: retain only vocal model presets/architectures
and remove training, validation, fine-tuning, GUI, and their dedicated code.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path

VOCAL_MODELS = (
    "mdx23c", "htdemucs", "segm_models", "torchseg", "mel_band_roformer",
    "mel_band_conformer", "bs_roformer", "bs_conformer", "bs_mamba2",
    "swin_upernet", "bandit", "scnet_unofficial", "scnet", "dttnet",
    "dbs_bsrnn", "dbs_bs_roformer", "dbs_mel_roformer",
    "mel_roformer_official", "bs_roformer_official", "vit_mel_roformer",
)

VOCAL_VIPERX_CONFIGS = {
    "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
}

EXCLUSIVE_MODEL_DIRS = (
    "models/bandit_v2/", "models/look2hear/", "models/scnet_roformer/",
    "models/sub_roformer/", "models/bandit/core/data/",
    "models/bandit/core/loss/", "models/bandit/core/metrics/",
    "models/bandit/core/utils/",
)
EXCLUSIVE_MODEL_FILES = {
    "models/bandit/model_from_config.py",  # Non-vocal DnR preset
    "models/conformer_model.py",           # No retained vocal preset
    "models/mdx23c_tfc_tdf_v3_with_STHT.py",
    "models/scnet/scnet_tran.py", "models/scnet/scnet_masked.py",
    "models/bs_roformer/bs_roformer_experimental.py",
    "models/bs_roformer/mel_band_roformer_experimental.py",
    "models/mel_band_conformer.py",        # Unreachable duplicate registry path
    "models/ex_bi_mamba2.py", "models/ts_bs_mamba2.py",
}

ROOT_KEEP = {
    ".gitignore", ".gitattributes", "LICENSE", "README.md",
    "inference.py", "pyproject.toml", "requirements.txt",
}
DOCS_KEEP = {"docs/pretrained_models.md", "docs/inference.md"}
TESTS_KEEP = {"tests/test_inference_only.py"}
UTILS_KEEP = {"utils/audio_utils.py", "utils/model_utils.py", "utils/settings.py"}


def extract_functions(source: str, names: tuple[str, ...], filename: str) -> str:
    """Extract unmodified function source, retaining implementation details."""
    tree = ast.parse(source, filename=filename)
    found = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = set(names) - found.keys()
    if missing:
        raise ValueError(f"{filename}: required inference functions absent: {sorted(missing)}")
    lines = source.splitlines(keepends=True)
    return "\n\n\n".join("".join(lines[found[name].lineno - 1 : found[name].end_lineno]).rstrip() for name in names) + "\n"


def build_model_utils(source: str) -> str:
    kept = extract_functions(source, (
        "compile_model", "is_compiled_model", "bigshifts_wrapper", "demix",
        "apply_tta", "_get_accumulation_device", "_getWindowingArray",
        "prefer_target_instrument", "strip_compile_prefix",
    ), "utils/model_utils.py")
    # torch.amp.autocast should never request CUDA on CPU/MPS inference.
    kept = kept.replace("with torch.cuda.amp.autocast(enabled=use_amp):",
                        "with torch.cuda.amp.autocast(enabled=(use_amp and torch.device(device).type == 'cuda')):")
    kept = kept.replace("Use torch.compile for faster inference and training", "Use torch.compile for faster inference")
    kept = kept.replace("needed for training", "for checkpoint compatibility")
    kept = kept.replace("during training validation the model fills its lazy caches", "the model may fill lazy caches")
    return '''"""Inference-only demixing, augmentation and checkpoint utilities."""
import argparse
import os
import shutil
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from ml_collections import ConfigDict
from tqdm.auto import tqdm


''' + kept + '''


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
'''


def build_audio_utils(source: str) -> str:
    kept = extract_functions(source, (
        "normalize_audio", "denormalize_audio", "draw_spectrogram",
    ), "utils/audio_utils.py")
    # Silent audio normalization must not produce NaNs.
    kept = kept.replace("return (audio - mean) / std, {\"mean\": mean, \"std\": std}",
                        "std = max(float(std), 1e-8)\n    return (audio - mean) / std, {\"mean\": mean, \"std\": std}")
    return '''"""Audio helpers used by the inference entry point."""
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


''' + kept


# Model constructors are retained from the upstream inference registry. These
# classes support vocal presets; some also have internal losses which cannot be
# removed without changing checkpoint/state-dict compatibility.
MODEL_REGISTRY = '''    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'mel_band_conformer':
        from models.bs_roformer import MelBandConformer
        model = MelBandConformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_conformer':
        from models.bs_roformer import BSConformer
        model = BSConformer(**dict(config.model))
    elif model_type == 'bs_mamba2':
        from models.bs_mamba2_code.bs_mamba2 import BSMamba2Model
        model = BSMamba2Model(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'dttnet':
        from models.DTTNet import DPTDFNet
        model = DPTDFNet(config)
    elif model_type == 'dbs_bsrnn':
        from models.dynamic_band_splitting import DBSBSRNN
        model = DBSBSRNN(**dict(config.model))
    elif model_type == 'dbs_bs_roformer':
        from models.dynamic_band_splitting import DBSBSRoformer
        model = DBSBSRoformer(**dict(config.model))
    elif model_type == 'dbs_mel_roformer':
        from models.dynamic_band_splitting import DBSMelRoformer
        model = DBSMelRoformer(**dict(config.model))
    elif model_type == 'mel_roformer_official':
        from models.BSRoFormer_official import MelRoFormer
        model = MelRoFormer(**dict(config.model))
    elif model_type == 'bs_roformer_official':
        from models.BSRoFormer_official import BSRoFormer
        model = BSRoFormer(**dict(config.model))
    elif model_type == 'vit_mel_roformer':
        from models.VitMelRoFormer import VitMelRoformer
        model = VitMelRoformer(**dict(config.model))
    else:
        raise ValueError(f'Unsupported vocal model architecture: {model_type}')
'''


def build_settings() -> str:
    return '''"""CLI arguments, model construction and vocal-only configuration checks."""
from __future__ import annotations

import argparse
from typing import Dict, Tuple, Union

import soundfile as sf
import torch
import yaml
from torch import nn
from omegaconf import OmegaConf
from ml_collections import ConfigDict

VOCAL_MODEL_TYPES = frozenset(%r)


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


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    if model_type == 'htdemucs':
        return OmegaConf.load(config_path)
    with open(config_path, encoding='utf-8') as f:
        # Trusted YAML presets may contain !!python/tuple used by upstream.
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, Union[ConfigDict, OmegaConf]]:
    config = load_config(model_type, config_path)
    validate_vocal_config(config)
    model_type = config.training.get('model_type', model_type)
    if model_type not in VOCAL_MODEL_TYPES:
        raise ValueError(f'Unsupported vocal model architecture: {model_type}')
''' % (VOCAL_MODELS,) + MODEL_REGISTRY + '''    return model, config
'''


def build_inference(source: str) -> str:
    # Preserve upstream inference handling, including bigshifts, GPU/MPS, TTA,
    # normalization and existing checkpoint format. Expose returned output paths.
    if 'def separate_folder(' in source and 'def main()' in source:
        return source  # already converted
    output = source
    if 'def run_folder(' not in output or 'def proc_folder(' not in output:
        raise ValueError('Unexpected inference.py: entry points not found')
    output = output.replace('    model.eval()\n', '    model.eval()\n    written_files = []\n', 1)
    output = output.replace('            sf.write(output_path, estimates.T, sr, subtype=subtype)\n',
                            '            sf.write(output_path, estimates.T, sr, subtype=subtype)\n            written_files.append(output_path)\n', 1)
    output = output.replace('    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")',
                            '    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")\n    return written_files', 1)
    output = output.replace('            if peak <= 1.0 and args.pcm_type != \'FLOAT\':\n                codec = "flac"\n            else:\n                codec = "wav"',
                            '            codec = "flac" if args.flac_file else "wav"', 1)
    if '            codec = "flac" if args.flac_file else "wav"' not in output:
        raise ValueError('Unexpected inference.py: output codec logic missing')
    output = output.replace('    run_folder(model, args, config, device, verbose=True)\n',
                            '    return run_folder(model, args, config, device, verbose=True)\n', 1)
    # Derive instrumental from denormalized vocals. The original upstream
    # block mixes normalized estimates and unnormalized mixtures.
    before = '''        if args.extract_instrumental:
            instr = "vocals" if "vocals" in instruments else instruments[0]
            waveforms_orig["instrumental"] = mix_orig - waveforms_orig[instr]
            if "instrumental" not in instruments:
                instruments.append("instrumental")'''
    after = '''        if args.extract_instrumental:
            instr = "vocals"
            vocals_estimate = waveforms_orig[instr]
            if config.inference.get("normalize", False):
                vocals_estimate = denormalize_audio(vocals_estimate, norm_params)
            waveforms_orig["instrumental"] = mix_orig - vocals_estimate
            if "instrumental" not in instruments:
                instruments.append("instrumental")'''
    if before not in output:
        raise ValueError('Unexpected inference.py: instrumental extraction block missing')
    output = output.replace(before, after, 1)
    output = output.replace('            if "normalize" in config.inference:\n'
                            '                if config.inference["normalize"] is True:\n'
                            '                    estimates = denormalize_audio(estimates, norm_params)',
                            '            if instr != "instrumental" and config.inference.get("normalize", False):\n'
                            '                estimates = denormalize_audio(estimates, norm_params)', 1)
    output = output.replace('            peak: float = float(np.abs(estimates).max())\n', '', 1)
    output = output.replace(') -> None:\n    """\n    Process a folder of audio files for source separation.',
                            ') -> list[str]:\n    """\n    Process a folder of audio files for source separation.', 1)
    output = output.replace('import warnings\n\nwarnings.filterwarnings("ignore")', '', 1)
    output = output.replace('if __name__ == "__main__":\n    proc_folder(None)', '''def separate_folder(*, model_type: str, config_path: str,
                    checkpoint_path: str, input_folder: str,
                    output_folder: str, **options) -> list[str]:
    """Run vocal separation from Python and return written audio paths.

    Example: separate_folder(model_type='bs_roformer',
        config_path='configs/config_vocals_bs_roformer.yaml',
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
    main()''')
    if 'def separate_folder' not in output or 'def main()' not in output:
        raise ValueError('Unexpected inference.py ending')
    # Prevent silent random-weight inference; this is also validated in args.
    return output


def build_pyproject() -> str:
    return '''[project]
name = "music-source-separation-training"
description = "Vocal-only checkpoint inference from the CLI or Python. No training, fine-tuning, or GUI."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Roman Solovyev (ZFTurbo)" }]
dependencies = [
  "PyYAML",
  "librosa",
  "matplotlib",
  "ml-collections",
  "numpy>=2.0.0",
  "omegaconf",
  "soundfile",
  "torch>=2.6.0",
  "torchaudio",
  "tqdm"
]
dynamic = ["version"]

[project.scripts]
vocal-separate = "inference:main"

[project.urls]
Repository = "https://github.com/ZFTurbo/Music-Source-Separation-Training"

[project.optional-dependencies]
# Only model-family dependencies; no train / validation / LoRA / GUI extras.
bs_roformer = ["beartype", "einops", "packaging", "rotary-embedding-torch", "PoPE-pytorch"]
mel_band_roformer = ["beartype", "einops", "packaging", "rotary-embedding-torch", "PoPE-pytorch"]
bs_conformer = ["beartype", "einops", "packaging", "rotary-embedding-torch"]
mel_band_conformer = ["beartype", "einops", "packaging", "rotary-embedding-torch"]
bs_mamba2 = ["beartype", "einops", "packaging", "mamba-ssm ; platform_system == 'Linux' and platform_machine == 'x86_64'", "rotary-embedding-torch"]
htdemucs = ["demucs", "einops", "openunmix"]
mdx23c = []
segm_models = ["segmentation-models-pytorch", "timm"]
torchseg = ["torchseg", "timm"]
swin_upernet = ["transformers"]
bandit = ["spafe"]
scnet_unofficial = ["beartype", "einops"]
scnet = []
dttnet = []
dbs_bsrnn = ["einops", "rotary-embedding-torch"]
dbs_bs_roformer = ["einops", "rotary-embedding-torch"]
dbs_mel_roformer = ["einops", "rotary-embedding-torch"]
mel_roformer_official = ["einops", "rotary-embedding-torch"]
bs_roformer_official = ["einops", "rotary-embedding-torch"]
vit_mel_roformer = ["einops", "rotary-embedding-torch"]

[build-system]
requires = ["setuptools>=80", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
include-package-data = true
py-modules = ["inference"]

[tool.setuptools.packages.find]
where = ["."]
include = ["models*", "utils*", "configs*"]
namespaces = true

[tool.setuptools.package-data]
configs = ["**/*.yaml"]
'''


def build_readme() -> str:
    models = ', '.join('`' + name + '`' for name in VOCAL_MODELS)
    return '''# Vocal separation — inference only

这是 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
的 **Vocal-only / inference-only** 精简版本。

- ✅ 保留：人声分离模型、预训练权重推理、命令行、Python API。
- ✅ 人声模型的 `other` / `instrumental` 指伴奏输出，不是非人声模型分类。
- ❌ 删除：训练、验证/基准测试、微调/LoRA、自动批大小训练、GUI、音轨集成、数据集准备脚本。
- ⚠️ 模型架构文件中的必要权重结构、前向传播和个别内部 loss 不能随意删除，
  以免破坏推理所需的模型实例和 checkpoint 兼容性。

## Install

Install a compatible PyTorch (CPU or CUDA) build first, then:

```bash
python -m pip install -e ".[mel_band_roformer]"
```

Choose the model-family extra corresponding to your pretrained model.
For example, `.[bs_roformer]`, `.[mdx23c]` or `.[htdemucs]`.
Some architectures need GPU/Linux-specific dependencies.

## Command-line inference

```bash
vocal-separate \\
  --model_type mel_band_roformer \\
  --config_path configs/config_vocals_mel_band_roformer.yaml \\
  --start_check_point /path/to/vocals.ckpt \\
  --input_folder input_audio \\
  --store_dir outputs
```

Without installation, use `python inference.py` with the same arguments.
To additionally output accompaniment from the predicted vocals, add
`--extract_instrumental`. For all available inference flags: `vocal-separate --help`.

## Python API

```python
from inference import separate_folder

files = separate_folder(
    model_type="mel_band_roformer",
    config_path="configs/config_vocals_mel_band_roformer.yaml",
    checkpoint_path="/path/to/vocals.ckpt",
    input_folder="input_audio",
    output_folder="outputs",
    extract_instrumental=True,
)
print(files)
```

Only **full, trusted** checkpoints are supported. Loading arbitrary PyTorch
checkpoints can execute untrusted pickle content; do not load unknown sources.

## Vocal presets

Models supported: ''' + models + '''.

See [pretrained Vocal models](docs/pretrained_models.md) and the
[detailed inference guide](docs/inference.md). No training features remain in
the public CLI or API.

Original upstream project and its license remain credited.
'''


def build_inference_doc() -> str:
    return '''# Vocal inference (CLI and Python only)

这个版本只支持人声分离推理，不提供训练、微调、GUI 或验证命令。

## 1. 安装

```bash
python -m pip install -e ".[mel_band_roformer]"
```

先安装与你的 GPU / CPU 匹配的 PyTorch。如果使用其他人声模型，安装对应
`pyproject.toml` 中的模型 extra，例如 `.[bs_roformer]`。

## 2. 下载人声模型权重和配置

从 [Vocal pretrained models](pretrained_models.md) 中选择匹配的
checkpoint 和配置。模型类型（`--model_type`）必须与配置匹配。
仓库内的 `configs/config_vocals_*.yaml` 和两个 viperx vocal 预设均保留。

## 3. 命令行

```bash
python inference.py \\
  --model_type bs_roformer \\
  --config_path configs/config_vocals_bs_roformer.yaml \\
  --start_check_point weights/vocals.ckpt \\
  --input_folder input_audio \\
  --store_dir outputs \\
  --extract_instrumental
```

安装后也可使用 `vocal-separate` 替换 `python inference.py`。
支持 CPU (`--force_cpu`)、TTA (`--use_tta`)、BigShifts (`--bigshifts`)、
音频格式 (`--flac_file --pcm_type PCM_24`)、输出命名和单 GPU 编译。
通过 `--help` 查看参数。

## 4. Python 代码

```python
from inference import separate_folder

written = separate_folder(
    model_type="bs_roformer",
    config_path="configs/config_vocals_bs_roformer.yaml",
    checkpoint_path="weights/vocals.ckpt",
    input_folder="input_audio",
    output_folder="outputs",
    extract_instrumental=True,
)
for file in written:
    print(file)
```

也可调用兼容入口 `inference.proc_folder({...})`。

## 模型兼容性

- `vocals` 是必须的输出/目标音轨，`other` 和 `instrumental` 仅可作为伴奏。
- `vocals` + `other` 两路输出仍是 Vocal 模型，不会被误删。
- 部分模型架构包含内部 loss 或依赖训练框架的基类，保留这些是为推理时
  构建网络和加载参数；此分支不提供训练调用入口。
- 此精简分支不提供 LoRA adapter 推理/微调，只接受完整权重。
- PyTorch checkpoint 可能含有 pickle，请仅加载可信来源的文件。
'''


def build_pretrained(source: str) -> str:
    vocal_match = re.search(r'(?m)^### Vocal models\s*$', source)
    if not vocal_match:
        raise ValueError('docs/pretrained_models.md has no Vocal models section')
    rest = source[vocal_match.start():]
    next_heading = re.search(r'(?m)^### (?!Vocal models\s*$)', rest[len('### Vocal models'):])
    if next_heading:
        rest = rest[:len('### Vocal models') + next_heading.start()]
    return ('# Pretrained vocal separation models\n\n'
            '仅列出 Vocal models；other / instrumental 是对应的伴奏音轨。\n'
            '训练代码已移除，表内模型和权重仅用于推理。\n\n' + rest.strip() + '\n')


def build_bandit_wrapper(source: str) -> str:
    """BandIt only used LightningModule as an nn.Module superclass.

    Upstream wrapper.py has no calls to training hooks or Lightning APIs;
    substituting nn.Module keeps forward/state_dict layout intact.
    """
    old_import = 'import pytorch_lightning as pl'
    old_super = 'class BandSplitWrapperBase(pl.LightningModule):'
    if 'class BandSplitWrapperBase(nn.Module):' in source and old_import not in source:
        return source
    if old_import not in source or old_super not in source:
        raise ValueError('Unexpected BandIt wrapper; inspect before removing Lightning')
    return source.replace(old_import, '').replace(
        old_super, 'class BandSplitWrapperBase(nn.Module):')


def build_test() -> str:
    return '''"""Regression checks for the inference-only vocal fork."""
from pathlib import Path
import ast
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_no_training_gui_or_finetuning_entrypoints():
    for file in ("train.py", "train_ddp.py", "train_accelerate.py",
                 "valid.py", "valid_ddp.py", "ensemble.py", "utils/dataset.py",
                 "utils/losses.py", "utils/auto_batch.py"):
        assert not (ROOT / file).exists(), file
    assert not (ROOT / "gui").exists()
    assert not (ROOT / "scripts").exists()


def test_inference_cli_exposes_no_training_or_lora_flags():
    from utils.settings import parse_args_inference
    with pytest.raises(ValueError, match="Unsupported inference options"):
        parse_args_inference({"train_lora_loralib": True})
    with pytest.raises(ValueError, match="Unsupported inference options"):
        parse_args_inference({"lora_checkpoint_loralib": "adapter.ckpt"})


def test_vocal_presets_only():
    from utils.settings import validate_vocal_config
    from ml_collections import ConfigDict
    presets = list((ROOT / "configs").rglob("*.yaml"))
    assert presets
    for path in presets:
        config = ConfigDict(yaml.load(path.read_text(), Loader=yaml.FullLoader))
        validate_vocal_config(config)
    for instruments in (["drums"], ["vocals", "bass"]):
        with pytest.raises(ValueError):
            validate_vocal_config(ConfigDict({"training": {"instruments": instruments}}))


def test_cli_help_and_python_api():
    result = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--model_type" in result.stdout
    assert "--config_path" in result.stdout
    assert "--start_check_point" in result.stdout
    assert "--train" not in result.stdout
    assert "--lora" not in result.stdout
    source = ast.parse((ROOT / "inference.py").read_text())
    assert {"proc_folder", "separate_folder", "main"}.issubset(
        {n.name for n in source.body if isinstance(n, ast.FunctionDef)})
'''


def should_delete(relative: str) -> bool:
    top = relative.split('/', 1)[0]
    if top in ("gui", "scripts"):
        return True
    if top == 'configs':
        return not (relative.endswith((".yaml", ".yml"))
                    and ('vocal' in relative.lower() or relative in VOCAL_VIPERX_CONFIGS))
    if top == 'docs':
        return relative not in DOCS_KEEP
    if top == 'tests':
        return relative not in TESTS_KEEP
    if top == 'utils':
        return relative not in UTILS_KEEP
    if top == 'models':
        return relative in EXCLUSIVE_MODEL_FILES or any(relative.startswith(d) for d in EXCLUSIVE_MODEL_DIRS)
    if '/' not in relative:
        return relative not in ROOT_KEEP
    # Preserve unexpected model dependencies rather than breaking imports.
    return False


def build_plan(root: Path) -> tuple[dict[Path, str], list[Path]]:
    required = ("inference.py", "utils/model_utils.py", "utils/audio_utils.py",
                "utils/settings.py", "docs/pretrained_models.md", "pyproject.toml")
    for rel in required:
        if not (root / rel).is_file():
            raise FileNotFoundError(f'Missing required upstream file: {rel}')
    new_contents = {
        'inference.py': build_inference((root / 'inference.py').read_text(encoding='utf-8')),
        'utils/model_utils.py': build_model_utils((root / 'utils/model_utils.py').read_text(encoding='utf-8')),
        'utils/audio_utils.py': build_audio_utils((root / 'utils/audio_utils.py').read_text(encoding='utf-8')),
        'utils/settings.py': build_settings(),
        'README.md': build_readme(),
        'pyproject.toml': build_pyproject(),
        'requirements.txt': '\n'.join([
            'PyYAML', 'librosa', 'matplotlib', 'ml-collections', 'numpy>=2.0.0',
            'omegaconf', 'soundfile', 'torch>=2.6.0', 'torchaudio', 'tqdm', ''
        ]),
        'docs/pretrained_models.md': build_pretrained((root / 'docs/pretrained_models.md').read_text(encoding='utf-8')),
        'docs/inference.md': build_inference_doc(),
        'tests/test_inference_only.py': build_test(),
    }
    bandit_wrapper = root / 'models/bandit/core/model/bsrnn/wrapper.py'
    if bandit_wrapper.exists():
        new_contents['models/bandit/core/model/bsrnn/wrapper.py'] = build_bandit_wrapper(
            bandit_wrapper.read_text(encoding='utf-8'))
    edits: dict[Path, str] = {}
    for rel, content in new_contents.items():
        path = root / rel
        if not path.is_file() or path.read_text(encoding='utf-8') != content:
            edits[path] = content
    deleted = []
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if '.git' in rel.parts or '__pycache__' in rel.parts:
            continue
        if should_delete(rel.as_posix()):
            deleted.append(path)
    for rel, content in edits.items():
        if rel.suffix == '.py':
            compile(content, str(rel), 'exec')
    return edits, sorted(set(deleted))


def clean_git(root: Path) -> None:
    if not (root / '.git').exists():
        raise ValueError('Run --apply only on a Git checkout, on a separate branch')
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(args, cwd=root, check=False, capture_output=True)
        if result.returncode:
            raise ValueError('Working tree has tracked changes; use a clean checkout')
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise ValueError('Working tree contains untracked files; clean checkout required')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--apply', action='store_true', help='Apply to clean feature-branch checkout')
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    if args.apply:
        clean_git(root)
    edits, deletes = build_plan(root)
    print(f'Vocal inference-only: {len(edits)} edits/new files, {len(deletes)} deletions')
    for p in sorted(edits):
        print('  EDIT  ', p.relative_to(root))
    for p in deletes:
        print('  DELETE', p.relative_to(root))
    if not args.apply:
        print('\nPREVIEW ONLY. Re-run with --apply after creating a feature branch.')
        return
    for path, contents in edits.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding='utf-8')
    for path in deletes:
        path.unlink()
    for d in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if '.git' not in d.parts and not any(d.iterdir()):
            d.rmdir()
    print('\nChanges applied locally. Review git diff and run python -m pytest tests/test_inference_only.py.')


if __name__ == '__main__':
    main()
