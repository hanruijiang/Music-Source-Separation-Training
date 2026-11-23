# Vocal inference (CLI and Python only)

这个版本只支持人声分离推理，不提供训练、微调、GUI 或验证命令。

## 1. 安装

```bash
python -m pip install -e ".[mel_band_roformer]"
```

先安装与你的 GPU / CPU 匹配的 PyTorch。如果使用其他人声模型，安装对应
`pyproject.toml` 中的模型 extra，例如 `.[bs_roformer]`。
BandIt 和官方 RoFormer 的 extra 会自动安装 `torchaudio`；其他模型不会强制安装。
启用 `--draw_spectro` 时再安装 `.[spectrogram]`；模型配置设置 `use_pope: true`
时再额外安装 `.[pope]`。

## 2. 下载人声模型权重和配置

从 [Vocal pretrained models](pretrained_models.md) 中选择匹配的
checkpoint 和配置。模型类型（`--model_type`）必须与配置匹配。
仓库内的 `configs/config_vocals_*.yaml` 和两个 viperx vocal 预设均保留。

## 3. 命令行

```bash
python inference.py \
  --model_type bs_roformer \
  --config_path configs/config_vocals_bs_roformer.yaml \
  --start_check_point weights/vocals.ckpt \
  --input_folder input_audio \
  --store_dir outputs \
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
- 推理所需的模型结构和 checkpoint 参数命名仍保留；训练框架的入口和
  独立训练测试脚本已移除。
- 此精简分支不提供 LoRA adapter 推理/微调，只接受完整权重。
- `scnet_unofficial` 保留预置的 RNN 推理结构，未保留未使用的 Mamba 变体。
- PyTorch checkpoint 可能含有 pickle，请仅加载可信来源的文件。
