# Five selected vocal models

只保留以下 5 个**具体版本**。SDR 为上游 Multisong Dataset 参考结果，
非本分支新测得的分数。

| 模型 | `model_type` | Vocal SDR | 本地配置 | 原始 checkpoint |
|---|---|---|---|---|
| MDX23C | `mdx23c` | 10.17 | [YAML](../configs/config_vocals_mdx23c.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt) |
| Segm Models (VitLarge23) | `segm_models` | 9.77 | [YAML](../configs/config_vocals_segm_models.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt) |
| BS Roformer (viperx) | `bs_roformer` | 10.87 | [YAML](../configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml) | [Weights](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt) |
| MelBand Roformer (KimberleyJensen) | `mel_band_roformer` | 10.98 | [YAML](../configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml) | [Weights](https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt) |
| BS PolarFormer (PoPE) | `bs_polarformer` | 11.00 | [YAML](../configs/model_bs_polarformer_float16.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.ckpt) |

BS PolarFormer: BS Roformer + PoPE，
[原始配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.20/model_bs_polarformer_float16.yaml)。
需安装 `.[bs_polarformer]`。
