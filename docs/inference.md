# 五个预训练人声模型的推理指南

项目仓库：[hanruijiang/Music-Source-Separation-Training](https://github.com/hanruijiang/Music-Source-Separation-Training)。
本仓库中的 YAML 和说明使用相对路径；部分 checkpoint 的下载地址保留在
上游 Release，因为本 fork 尚未发布对应的权重文件。

仅保留命令行 `python inference.py` / `vocal-separate` 和 Python
API `inference.separate_folder`，不包含训练或 GUI。

| 模型 | `--model_type` | 配置 |
|---|---|---|
| MDX23C | `mdx23c` | `configs/config_vocals_mdx23c.yaml` |
| Segm Models (VitLarge23) | `segm_models` | `configs/config_vocals_segm_models.yaml` |
| BS Roformer (viperx) | `bs_roformer` | `configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml` |
| MelBand Roformer (KimberleyJensen) | `mel_band_roformer` | `configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml` |
| BS PolarFormer (PoPE) | `bs_polarformer` | `configs/model_bs_polarformer_float16.yaml` |

安装时选择与预训练权重对应的 extra，如
`python -m pip install -e ".[bs_polarformer,test]"`。
`bs_polarformer` 是启用 `model.use_pope: true` 的 BS Roformer，
专用 extra 会安装 `PoPE-pytorch`。绘制频谱图需额外安装
`.[spectrogram]`。

## 命令行示例

```bash
vocal-separate \
  --model_type bs_roformer \
  --config_path configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --start_check_point /path/to/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --input_folder input_audio \
  --store_dir outputs \
  --extract_instrumental
```

支持 `--force_cpu`、`--use_tta`、`--bigshifts`、
`--flac_file`、`--pcm_type`、`--draw_spectro` 和 `--compile`。

## Python API

```python
from inference import separate_folder

written = separate_folder(
    model_type="bs_polarformer",
    config_path="configs/model_bs_polarformer_float16.yaml",
    checkpoint_path="/path/to/model_bs_polarformer_float16.ckpt",
    input_folder="audio",
    output_folder="results",
)
```

本地 YAML 仅保留推理必需的音频、模型、输出音轨、AMP 和推理参数。
请勿随意修改 `model` 参数，以免预训练权重不兼容。
只使用可信来源的完整 checkpoint；尚需进行真实音频和权重的端到端测试。
