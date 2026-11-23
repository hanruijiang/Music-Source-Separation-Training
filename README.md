# Vocal separation — inference only

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
For example, `.[bs_roformer]`, `.[mdx23c]` or `.[segm_models]`.
Some architectures need GPU/Linux-specific dependencies.

## Command-line inference

```bash
vocal-separate \
  --model_type mel_band_roformer \
  --config_path configs/config_vocals_mel_band_roformer.yaml \
  --start_check_point /path/to/vocals.ckpt \
  --input_folder input_audio \
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

Models supported: `mdx23c`, `segm_models`, `torchseg`, `mel_band_roformer`, `mel_band_conformer`, `bs_roformer`, `bs_conformer`, `bandit`, `scnet_unofficial`, `scnet`, `dttnet`, `dbs_bsrnn`, `dbs_bs_roformer`, `dbs_mel_roformer`, `mel_roformer_official`, `bs_roformer_official`, `vit_mel_roformer`.

See [pretrained Vocal models](docs/pretrained_models.md) and the
[detailed inference guide](docs/inference.md). No training features remain in
the public CLI or API.

Original upstream project and its license remain credited.
