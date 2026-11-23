"""Regression checks for the inference-only vocal fork."""
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


def test_removed_model_families_and_configs():
    from utils.settings import VOCAL_MODEL_TYPES

    removed = {"htdemucs", "swin_upernet", "bs_mamba2"}
    assert not removed.intersection(VOCAL_MODEL_TYPES)
    for path in (
        "models/demucs4ht.py",
        "models/upernet_swin_transformers.py",
        "models/bs_mamba2_code/attend_mamba.py",
        "models/bs_mamba2_code/bs_mamba2.py",
        "configs/config_vocals_htdemucs.yaml",
        "configs/config_vocals_swin_upernet.yaml",
        "configs/config_bs_mamba2_vocals.yaml",
        "configs/config_vocals_bs_mamba2.yaml",
    ):
        assert not (ROOT / path).exists(), path

    settings = (ROOT / "utils/settings.py").read_text()
    demix = (ROOT / "utils/model_utils.py").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    assert not any(name in settings for name in removed)
    assert "htdemucs" not in demix
    assert not any(f"{name} = " in project for name in removed)
    assert "omegaconf" not in project


def test_bandit_core_does_not_import_training_frameworks():
    import importlib

    core = importlib.import_module("models.bandit.core")
    assert not hasattr(core, "LightningSystem")
    source = (ROOT / "models/bandit/core/__init__.py").read_text()
    for obsolete in ("pytorch_lightning", "torchmetrics", "asteroid", "core.data"):
        assert obsolete not in source
