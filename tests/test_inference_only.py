"""Contract tests for the five retained pretrained vocal models."""
from pathlib import Path
import ast
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "mdx23c": "configs/config_vocals_mdx23c.yaml",
    "segm_models": "configs/config_vocals_segm_models.yaml",
    "bs_roformer": "configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "mel_band_roformer": "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
    "bs_polarformer": "configs/model_bs_polarformer_float16.yaml",
}
MODEL_FILES = {
    "mdx23c_tfc_tdf_v3.py", "segm_models.py",
    "bs_roformer/__init__.py", "bs_roformer/attend.py",
    "bs_roformer/bs_roformer.py", "bs_roformer/mel_band_roformer.py",
}


def test_only_five_model_types():
    from utils.settings import VOCAL_MODEL_TYPES
    assert VOCAL_MODEL_TYPES == frozenset(MODEL_CONFIGS)


def test_only_five_checkpoint_presets_and_needed_modules():
    presets = {str(p.relative_to(ROOT)) for p in (ROOT / "configs").rglob("*.yaml")}
    model_files = {str(p.relative_to(ROOT / "models"))
                   for p in (ROOT / "models").rglob("*.py")}
    assert presets == set(MODEL_CONFIGS.values())
    assert model_files == MODEL_FILES


@pytest.mark.parametrize("model_type", list(MODEL_CONFIGS))
def test_pretrained_configs_have_only_inference_metadata(model_type):
    from utils.settings import load_config, validate_vocal_config
    config = load_config(str(ROOT / MODEL_CONFIGS[model_type]))
    validate_vocal_config(config)
    assert set(config) == {"audio", "model", "training", "inference"}
    assert set(config.training) == {"instruments", "target_instrument", "use_amp"}
    assert "vocals" in config.training.instruments
    assert config.inference.num_overlap > 0


def test_polarformer_is_bs_roformer_with_pope():
    from utils.settings import load_config
    config = load_config(str(ROOT / MODEL_CONFIGS["bs_polarformer"]))
    model = config.model
    assert model.use_pope is True
    assert model.dim == 256 and model.depth == 12
    assert model.stft_hop_length == 512
    assert model.num_stems == 1
    assert sum(model.freqs_per_bands) == model.dim_freqs_in == 1025
    src = (ROOT / "utils/settings.py").read_text()
    assert '"bs_roformer", "bs_polarformer"' in src
    assert "BS PolarFormer requires model.use_pope: true" in src


def test_only_required_model_extras():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = set(project["project"]["optional-dependencies"])
    assert extras == {*MODEL_CONFIGS, "spectrogram", "test"}
    assert "PoPE-pytorch" in project["project"]["optional-dependencies"]["bs_polarformer"]
    assert "PoPE-pytorch" not in project["project"]["optional-dependencies"]["bs_roformer"]


def test_no_online_imagenet_download_in_segm():
    src = (ROOT / "models/segm_models.py").read_text()
    assert 'encoder_weights="imagenet"' not in src
    assert "encoder_weights=None" in src


def test_cli_and_api():
    from utils.settings import parse_args_inference
    with pytest.raises(ValueError, match="Unsupported inference options"):
        parse_args_inference({"train_lora_loralib": True})
    completed = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--help"],
                               capture_output=True, text=True, check=True)
    for model_type in MODEL_CONFIGS:
        assert model_type in completed.stdout
    tree = ast.parse((ROOT / "inference.py").read_text())
    func_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"proc_folder", "separate_folder", "main"} <= func_names


def test_no_training_or_gui():
    for path in ("train.py", "valid.py", "ensemble.py", "train_ddp.py", "gui", "scripts"):
        assert not (ROOT / path).exists()


def test_repository_links_point_to_this_fork():
    repository = "https://github.com/hanruijiang/Music-Source-Separation-Training"
    readme = (ROOT / "README.md").read_text()
    docs = (ROOT / "docs/inference.md").read_text()
    assert f"[hanruijiang/Music-Source-Separation-Training]({repository})" in readme
    assert repository in docs
    # Keep original author attribution and verified upstream weight downloads.
    old = "https://github.com/ZFTurbo/Music-Source-Separation-Training"
    assert f"{old}/releases/download/" in readme
    assert f"Repository = \"{repository}\"" in (ROOT / "pyproject.toml").read_text()
