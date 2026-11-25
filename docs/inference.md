# 五个预训练人声模型的推理指南

项目仓库：[hanruijiang/Music-Source-Separation-Training](https://github.com/hanruijiang/Music-Source-Separation-Training)。
推理接口包括命令行 `python inference.py` / `vocal-separate` 和 Python API `inference.separate_folder`。

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
默认输出保留输入的相对目录（`{dir_name}/{file_name}/{instr}`），
同一次运行中若输出路径冲突会报错；可用 `--filename_template` 指定不同模板。
若同一目录有同名不同扩展名的输入，可使用 `{dir_name}/{source_name}/{instr}`（`source_name` 包含扩展名）。
重复运行仍可覆盖此前的结果。输入目录无可处理音频时会报错。

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

YAML 配置包含音频、模型、输出音轨、AMP 和推理参数。
使用与配置匹配的预训练权重，并从可信来源获取 checkpoint。
