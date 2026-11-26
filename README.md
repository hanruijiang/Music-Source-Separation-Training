# Vocal separation

这是 [hanruijiang/Music-Source-Separation-Training](https://github.com/hanruijiang/Music-Source-Separation-Training)
提供以下五个预训练人声分离模型（四种底层架构）的推理接口。

| 模型版本 | `--model_type` | 配置 | 权重 |
|---|---|---|---|
| MDX23C | `mdx23c` | [YAML](configs/config_vocals_mdx23c.yaml) | [Checkpoint](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt) |
| Segm Models (VitLarge23) | `segm_models` | [YAML](configs/config_vocals_segm_models.yaml) | [Checkpoint](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt) |
| BS Roformer (viperx) | `bs_roformer` | [YAML](configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml) | [Checkpoint](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt) |
| MelBand Roformer (KimberleyJensen) | `mel_band_roformer` | [YAML](configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml) | [Checkpoint](https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt) |
| BS PolarFormer (PoPE) | `bs_polarformer` | [YAML](configs/model_bs_polarformer_float16.yaml) | [Checkpoint](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.ckpt) |

**BS PolarFormer** 通过 `use_pope: true` 复用 `BSRoformer` 架构，并启用
PoPE（Polar Positional Embeddings）。
`vocals` 是人声，`other` 是伴奏。部分模型仅直接输出人声，
可用 `--extract_instrumental` 从混音中减去人声获得伴奏。

## 安装

先安装合适的 CPU/CUDA PyTorch，随后按需安装**其中一种**模型依赖：

```bash
python -m pip install -e ".[mdx23c]"
python -m pip install -e ".[segm_models]"
python -m pip install -e ".[bs_roformer]"
python -m pip install -e ".[mel_band_roformer]"
python -m pip install -e ".[bs_polarformer]"
```

`bs_polarformer` extra 自动包含 `PoPE-pytorch`，其它模型不会安装它。
使用 `--draw_spectro` 时额外安装 `.[spectrogram]`。

## CLI

```bash
python inference.py \
  --model_type bs_polarformer \
  --config_path configs/model_bs_polarformer_float16.yaml \
  --start_check_point /path/to/model_bs_polarformer_float16.ckpt \
  --input_folder input_audio \
  --store_dir outputs \
  --extract_instrumental
```

安装后可用 `vocal-separate` 代替 `python inference.py`。

可选推理前模型优化：`--optimization safe`（默认关闭），可与独立的
`--compile` 结合使用。详情及限制见[推理指南](docs/inference.md#可选推理前模型优化)。

## Python API

```python
from inference import separate_folder

paths = separate_folder(
    model_type="mel_band_roformer",
    config_path="configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
    checkpoint_path="/path/to/MelBandRoformer.ckpt",
    input_folder="audio",
    output_folder="outputs",
    extract_instrumental=True,
)
print(paths)
```

### 独立评分 API

```python
from utils.metrics import score_tracks

# 计算 SDR（dB）。
scores = score_tracks(reference, estimate)

# 计算四项指标。
scores = score_tracks(
    reference, estimate,
    metrics=("sdr", "si_sdr", "bleedless", "fullness"),
    sample_rate=44100,
)
```

SDR、SI-SDR 仅依赖 NumPy；Bleedless、Fullness 需要 PyTorch、Librosa
及与 PyTorch 版本和 CPU/CUDA 构建匹配的 Torchaudio，可安装 `.[metrics]`。
输入为形状相同的非空 `[channels, samples]` 波形，采样率、幅度尺度一致，
且所有样本均为有限数值。默认使用 CPU；频谱评分可通过 `device="cuda"` 执行。
`sample_rate` 为音轨的实际采样率。评分独立于推理计时。

仅加载可信的完整 PyTorch checkpoint；加载恶意 pickle 权重具有执行代码风险。

模型权重来源见上方表格。更多细节见 [推理指南](docs/inference.md)、[模型与权重](docs/pretrained_models.md)。
