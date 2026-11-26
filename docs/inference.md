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

## 可选推理前模型优化

使用 `--optimization safe`（Python API: `optimization="safe"`）启用实验性
等价优化。默认 `off`，便于与原始推理比较。先严格加载原始 checkpoint，
再 eval、转换模型、移动设备，最后按需 `--compile`。优化后的内存模型
不作为新的 checkpoint 格式保存。

- BS Roformer / Mel Band Roformer：QKV + Gate 投影合并。
- BS PolarFormer：保留独立 QKV / Gate 投影；在 eval、无梯度、CUDA、
  无掩码、非因果、零 dropout 且末维 stride 为 1 的标准自注意力中，
  使用 PoPE-pytorch 0.2.1 已核对的 Triton forward launcher，跳过 Q/K/V
  布局复制。其他版本、无 Triton 或不满足条件时仍走原 PoPE API。
- 三个 Roformer：按频带形状分组的 BandSplit / MaskEstimator 批量 GEMM
  （每组最多四个频带）、默认 Hann window 缓存。
  保留原始 F.normalize、Tanh、GLU 和频带顺序。
- Segm Models：仅融合真实 Sequential forward 中确认相邻的标准 Conv2d +
  BatchNorm2d；其他第三方结构不猜测融合。
- MDX23C / Segm Models：单设备 STFT window 缓存。
- 普通 Dropout 替换为 Identity。动态 InstanceNorm、RMSNorm 和非线性残差保留。
- `advanced` 在上述基础上额外尝试单 CUDA GPU 卷积模型 channels-last 权重布局。

`--compile` 仍独立控制。自定义归一化 kernel、PoPE/RoPE 缓存、
Linear-BN 等未确认安全的转换暂不启用。数学等价不保证浮点逐位一致，
实际性能和峰值显存取决于 GPU、AMP、批大小和输入长度；尚未完成五个
预训练模型的 GPU 性能及音频质量验证，故优化保持 opt-in。若不兼容或
速度下降，使用 `--optimization off` 回退。请勿将优化后的 state_dict
当作原始 checkpoint 加载。
