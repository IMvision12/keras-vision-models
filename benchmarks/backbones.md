# Backbones — Per-Variant Top-1 & Params

ImageNet-1K Top-1 accuracy and parameter counts for **every backbone variant in kmodels** (187 variants across 37 families). Variants are listed in increasing parameter order within each family.

Two Top-1 columns are provided:

- **Paper Top-1 (%)** — the ImageNet-1K result reported in the **original publication's main results table**, in1k-pretrained, single-crop, at the reference resolution. `—` means the paper doesn't include that exact variant (e.g. `ConvNeXtAtto`/`Femto`/`Pico`/`Nano` were added later by timm and aren't in the ConvNeXt paper) **or** the value is not reliably documented in a single canonical table. PRs to fill these in are welcome.
- **Best Top-1 (%)** — the highest Top-1 across **all validated public checkpoints** for that exact architecture, taken from the [official timm ImageNet-1K results CSV](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv). When a variant has multiple recipes (in1k, in21k_ft_in1k, distilled, larger crop, etc.), the highest-scoring one wins. The `Source` column shows the exact timm checkpoint id so you can cross-reference.

The `Best` value is generally **higher** than the `Paper` value because most architectures got new recipes after publication. For example, ResNet-50 went from ~76% in the He et al. paper to 81.2% with timm's `a1_in1k` recipe.

> **Use which?** If you want to know how the model will perform when you load it via `kmodels` (which downloads timm weights), use the **Best** column. If you want the architecture's "headline" published number for a paper comparison, use **Paper**.

---

### CaiT &mdash; [paper](https://arxiv.org/abs/2103.17239)

Paper baselines: Table 5 of Touvron et al. 2021 (without distillation, in1k-only).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `CaiTXXS24` | 78.4 | 81.0 | 12 | `cait_xxs24_384.fb_dist_in1k` |
| `CaiTXXS36` | 79.7 | 82.2 | 17 | `cait_xxs36_384.fb_dist_in1k` |
| `CaiTXS24` | 81.8 | 84.1 | 27 | `cait_xs24_384.fb_dist_in1k` |
| `CaiTS24` | 82.7 | 85.0 | 47 | `cait_s24_384.fb_dist_in1k` |
| `CaiTS36` | 83.3 | 85.5 | 68 | `cait_s36_384.fb_dist_in1k` |
| `CaiTM36` | 83.8 | 86.1 | 271 | `cait_m36_384.fb_dist_in1k` |
| `CaiTM48` | 84.7 | 86.5 | 356 | `cait_m48_448.fb_dist_in1k` |

### ConvMixer &mdash; [paper](https://arxiv.org/abs/2201.09792)

Paper baselines: Table 1 of Trockman & Kolter 2022.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ConvMixer768D32` | 80.2 | 80.2 | 21 | `convmixer_768_32.in1k` |
| `ConvMixer1024D20` | 76.9 | 76.9 | 24 | `convmixer_1024_20_ks9_p14.in1k` |
| `ConvMixer1536D20` | 81.4 | 81.4 | 52 | `convmixer_1536_20.in1k` |

### ConvNeXt &mdash; [paper](https://arxiv.org/abs/2201.03545)

Paper baselines: Table 2 of Liu et al. 2022 (in1k-only, 224, single-crop). The Atto/Femto/Pico/Nano sizes are timm additions and not in the paper.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ConvNeXtAtto` | — | 77.0 | 3.7 | `convnext_atto.d2_in1k` |
| `ConvNeXtFemto` | — | 78.7 | 5.2 | `convnext_femto.d1_in1k` |
| `ConvNeXtPico` | — | 80.4 | 9.1 | `convnext_pico.d1_in1k` |
| `ConvNeXtNano` | — | 83.3 | 16 | `convnext_nano.r384_in12k_ft_in1k` |
| `ConvNeXtTiny` | 82.1 | 85.2 | 29 | `convnext_tiny.in12k_ft_in1k_384` |
| `ConvNeXtSmall` | 83.1 | 86.2 | 50 | `convnext_small.in12k_ft_in1k_384` |
| `ConvNeXtBase` | 83.8 | 87.1 | 89 | `convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384` |
| `ConvNeXtLarge` | 84.3 | 87.5 | 198 | `convnext_large.fb_in22k_ft_in1k_384` |
| `ConvNeXtXLarge` | — | 87.8 | 350 | `convnext_xlarge.fb_in22k_ft_in1k_384` |

### ConvNeXt V2 &mdash; [paper](https://arxiv.org/abs/2301.00808)

Paper baselines: Table 5 of Woo et al. 2023 (FCMAE pretrained → in1k fine-tuned, single-crop, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ConvNeXtV2Atto` | 76.7 | 77.8 | 3.7 | `convnextv2_atto.fcmae_ft_in1k` |
| `ConvNeXtV2Femto` | 78.5 | 79.3 | 5.2 | `convnextv2_femto.fcmae_ft_in1k` |
| `ConvNeXtV2Pico` | 80.3 | 81.1 | 9.1 | `convnextv2_pico.fcmae_ft_in1k` |
| `ConvNeXtV2Nano` | 81.9 | 83.4 | 16 | `convnextv2_nano.fcmae_ft_in22k_in1k_384` |
| `ConvNeXtV2Tiny` | 83.0 | 85.1 | 29 | `convnextv2_tiny.fcmae_ft_in22k_in1k_384` |
| `ConvNeXtV2Base` | 84.9 | 87.6 | 89 | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| `ConvNeXtV2Large` | 85.8 | 88.2 | 198 | `convnextv2_large.fcmae_ft_in22k_in1k_384` |
| `ConvNeXtV2Huge` | 86.3 | 88.9 | 660 | `convnextv2_huge.fcmae_ft_in22k_in1k_512` |

### DeiT &mdash; [paper](https://arxiv.org/abs/2012.12877)

Paper baselines: Table 6 of Touvron et al. 2021 (in1k-only, 224 unless noted).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `DEiTTiny16` | 72.2 | 72.2 | 5.7 | `deit_tiny_patch16_224.fb_in1k` |
| `DEiTTinyDistilled16` | 74.5 | 74.5 | 5.9 | `deit_tiny_distilled_patch16_224.fb_in1k` |
| `DEiTSmall16` | 79.8 | 79.9 | 22 | `deit_small_patch16_224.fb_in1k` |
| `DEiTSmallDistilled16` | 81.2 | 81.2 | 22 | `deit_small_distilled_patch16_224.fb_in1k` |
| `DEiTBase16` | 81.8 | 83.1 | 87 | `deit_base_patch16_384.fb_in1k` |
| `DEiTBaseDistilled16` | 83.4 | 85.4 | 88 | `deit_base_distilled_patch16_384.fb_in1k` |

### DeiT III &mdash; [paper](https://arxiv.org/abs/2204.07118)

Paper baselines: Table 4 of Touvron et al. 2022 (in1k from scratch, 224 unless noted).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `DEiT3Small16` | 81.4 | 84.8 | 22 | `deit3_small_patch16_384.fb_in22k_ft_in1k` |
| `DEiT3Medium16` | 82.6 | 84.5 | 39 | `deit3_medium_patch16_224.fb_in22k_ft_in1k` |
| `DEiT3Base16` | 83.8 | 86.7 | 87 | `deit3_base_patch16_384.fb_in22k_ft_in1k` |
| `DEiT3Large16` | 84.9 | 87.7 | 305 | `deit3_large_patch16_384.fb_in22k_ft_in1k` |
| `DEiT3Huge14` | 85.1 | 87.2 | 632 | `deit3_huge_patch14_224.fb_in22k_ft_in1k` |

### DenseNet &mdash; [paper](https://arxiv.org/abs/1608.06993)

Paper baselines: Table 2 of Huang et al. 2017 (single-crop, single-model on ImageNet val).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `DenseNet121` | 75.0 | 76.5 | 8.0 | `densenet121.ra_in1k` |
| `DenseNet169` | 76.2 | 75.9 | 14 | `densenet169.tv_in1k` |
| `DenseNet201` | 77.4 | 77.3 | 20 | `densenet201.tv_in1k` |
| `DenseNet161` | 77.7 | 77.4 | 29 | `densenet161.tv_in1k` |

### EfficientFormer &mdash; [paper](https://arxiv.org/abs/2206.01191)

Paper baselines: Table 1 of Li et al. 2022 (in1k from scratch, no distillation).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `EfficientFormerL1` | 79.2 | 80.5 | 12 | `efficientformer_l1.snap_dist_in1k` |
| `EfficientFormerL3` | 82.4 | 82.6 | 31 | `efficientformer_l3.snap_dist_in1k` |
| `EfficientFormerL7` | 83.3 | 83.4 | 82 | `efficientformer_l7.snap_dist_in1k` |

### EfficientNet &mdash; [paper](https://arxiv.org/abs/1905.11946)

Paper baselines: Table 2 of Tan & Le 2019 (single-crop, single-model). B8 and L2 come from the [Noisy Student](https://arxiv.org/abs/1911.04252) paper.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `EfficientNetB0` | 77.3 | 79.4 | 5.3 | `efficientnet_b0.ra4_e3600_r224_in1k` |
| `EfficientNetB1` | 79.2 | 81.4 | 7.8 | `efficientnet_b1.ra4_e3600_r240_in1k` |
| `EfficientNetB2` | 80.3 | 82.4 | 9.1 | `tf_efficientnet_b2.ns_jft_in1k` |
| `EfficientNetB3` | 81.7 | 84.1 | 12 | `tf_efficientnet_b3.ns_jft_in1k` |
| `EfficientNetB4` | 83.0 | 85.2 | 19 | `tf_efficientnet_b4.ns_jft_in1k` |
| `EfficientNetB5` | 83.7 | 86.1 | 30 | `tf_efficientnet_b5.ns_jft_in1k` |
| `EfficientNetB6` | 84.0 | 86.5 | 43 | `tf_efficientnet_b6.ns_jft_in1k` |
| `EfficientNetB7` | 84.4 | 86.8 | 66 | `tf_efficientnet_b7.ns_jft_in1k` |
| `EfficientNetB8` | 85.5 | 85.4 | 87 | `tf_efficientnet_b8.ap_in1k` |
| `EfficientNetL2` | 88.4 | 88.4 | 480 | `tf_efficientnet_l2.ns_jft_in1k` |

### EfficientNet-Lite &mdash; [paper](https://arxiv.org/abs/1905.11946)

Lite variants come from the [TensorFlow EfficientNet-Lite blog post / repo](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html). No formal paper baseline.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `EfficientNetLite0` | 75.1 | 74.8 | 4.7 | `tf_efficientnet_lite0.in1k` |
| `EfficientNetLite1` | 76.7 | 76.7 | 5.4 | `tf_efficientnet_lite1.in1k` |
| `EfficientNetLite2` | 77.6 | 77.5 | 6.1 | `tf_efficientnet_lite2.in1k` |
| `EfficientNetLite3` | 79.8 | 79.8 | 8.2 | `tf_efficientnet_lite3.in1k` |
| `EfficientNetLite4` | 81.5 | 81.5 | 13 | `tf_efficientnet_lite4.in1k` |

### EfficientNetV2 &mdash; [paper](https://arxiv.org/abs/2104.00298)

Paper baselines: Table 7 of Tan & Le 2021 (in1k-from-scratch baselines for S/M/L; B-series use the EfficientNetV1 conventions). XL is in21k-pretrained only.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `EfficientNetV2B0` | 78.7 | 78.4 | 7.1 | `tf_efficientnetv2_b0.in1k` |
| `EfficientNetV2B1` | 79.8 | 79.5 | 8.1 | `tf_efficientnetv2_b1.in1k` |
| `EfficientNetV2B2` | 80.5 | 80.2 | 10 | `tf_efficientnetv2_b2.in1k` |
| `EfficientNetV2B3` | 82.1 | 82.6 | 14 | `tf_efficientnetv2_b3.in21k_ft_in1k` |
| `EfficientNetV2S` | 83.9 | 84.3 | 21 | `tf_efficientnetv2_s.in21k_ft_in1k` |
| `EfficientNetV2M` | 85.1 | 86.0 | 54 | `tf_efficientnetv2_m.in21k_ft_in1k` |
| `EfficientNetV2L` | 85.7 | 86.8 | 119 | `tf_efficientnetv2_l.in21k_ft_in1k` |
| `EfficientNetV2XL` | — | 86.8 | 208 | `tf_efficientnetv2_xl.in21k_ft_in1k` |

### FlexiViT &mdash; [paper](https://arxiv.org/abs/2212.08013)

Paper baselines: Table 1 of Beyer et al. 2022 (in1k 1200ep, average of 16/32 patch sizes).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `FlexiViTSmall` | 81.4 | 82.6 | 22 | `flexivit_small.1200ep_in1k` |
| `FlexiViTBase` | 84.0 | 84.7 | 87 | `flexivit_base.1200ep_in1k` |
| `FlexiViTLarge` | 85.6 | 85.6 | 304 | `flexivit_large.1200ep_in1k` |

### Inception-ResNet-v2 &mdash; [paper](https://arxiv.org/abs/1602.07261)

Paper baseline: Szegedy et al. 2016 Table 5 (single-model, single-crop).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `InceptionResNetV2` | 80.4 | 80.4 | 56 | `inception_resnet_v2.tf_in1k` |

### Inception-v3 &mdash; [paper](https://arxiv.org/abs/1512.00567)

Paper baseline: Szegedy et al. 2015 Table 3 (single-model, single-crop).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `InceptionV3` | 78.8 | 78.8 | 24 | `inception_v3.gluon_in1k` |

### Inception-v4 &mdash; [paper](https://arxiv.org/abs/1602.07261)

Paper baseline: Szegedy et al. 2016 Table 5.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `InceptionV4` | 80.0 | 80.1 | 43 | `inception_v4.tf_in1k` |

### InceptionNeXt &mdash; [paper](https://arxiv.org/abs/2303.16900)

Paper baselines: Table 4 of Yu et al. 2023 (in1k 300ep, single-crop, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `InceptionNeXtAtto` | — | 75.3 | 4.2 | `inception_next_atto.sail_in1k` |
| `InceptionNeXtTiny` | 82.3 | 82.5 | 28 | `inception_next_tiny.sail_in1k` |
| `InceptionNeXtSmall` | 83.5 | 83.6 | 49 | `inception_next_small.sail_in1k` |
| `InceptionNeXtBase` | 84.0 | 85.2 | 87 | `inception_next_base.sail_in1k_384` |

### MaxViT &mdash; [paper](https://arxiv.org/abs/2204.01697)

Paper baselines: Table 2 of Tu et al. 2022 (in1k from scratch, 224, single-crop).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MaxViTTiny` | 83.6 | 85.7 | 31 | `maxvit_tiny_tf_512.in1k` |
| `MaxViTSmall` | 84.4 | 86.1 | 69 | `maxvit_small_tf_512.in1k` |
| `MaxViTBase` | 84.9 | 88.2 | 120 | `maxvit_base_tf_512.in21k_ft_in1k` |
| `MaxViTLarge` | 85.2 | 88.2 | 212 | `maxvit_large_tf_512.in21k_ft_in1k` |
| `MaxViTXLarge` | — | 88.5 | 476 | `maxvit_xlarge_tf_512.in21k_ft_in1k` |

### MiT &mdash; [paper](https://arxiv.org/abs/2105.15203)

Paper baselines: Table 5 of Xie et al. 2021 (SegFormer) — ImageNet-1K pretraining, single-crop. (timm doesn't validate MiT as a standalone classifier.)

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MiT_B0` | 70.5 | 70.5 | 3.7 | SegFormer paper |
| `MiT_B1` | 78.7 | 78.7 | 14 | SegFormer paper |
| `MiT_B2` | 81.6 | 81.6 | 25 | SegFormer paper |
| `MiT_B3` | 83.1 | 83.1 | 45 | SegFormer paper |
| `MiT_B4` | 83.6 | 83.6 | 63 | SegFormer paper |
| `MiT_B5` | 83.8 | 83.8 | 81 | SegFormer paper |

### MLP-Mixer &mdash; [paper](https://arxiv.org/abs/2105.01601)

Paper baselines: Table 1 of Tolstikhin et al. 2021 (in1k from scratch).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MLPMixerB16` | 76.4 | 82.3 | 60 | `mixer_b16_224.miil_in21k_ft_in1k` |
| `MLPMixerL16` | 71.8 | 72.1 | 208 | `mixer_l16_224.goog_in21k_ft_in1k` |

### MobileNetV2 &mdash; [paper](https://arxiv.org/abs/1801.04381)

Paper baselines: Table 4 of Sandler et al. 2018 (single-crop, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MobileNetV2WM50` | 65.4 | 65.9 | 2.0 | `mobilenetv2_050.lamb_in1k` |
| `MobileNetV2WM100` | 72.0 | 72.9 | 3.5 | `mobilenetv2_100.ra_in1k` |
| `MobileNetV2WM110` | — | 75.1 | 4.5 | `mobilenetv2_110d.ra_in1k` |
| `MobileNetV2WM120` | — | 77.3 | 5.8 | `mobilenetv2_120d.ra_in1k` |
| `MobileNetV2WM140` | 74.7 | 76.5 | 6.1 | `mobilenetv2_140.ra_in1k` |

### MobileNetV3 &mdash; [paper](https://arxiv.org/abs/1905.02244)

Paper baselines: Table 3 of Howard et al. 2019 (224, single-crop).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MobileNetV3SmallMinimal100` | 61.9 | 62.9 | 2.0 | `tf_mobilenetv3_small_minimal_100.in1k` |
| `MobileNetV3Small075` | 65.4 | 65.7 | 2.0 | `tf_mobilenetv3_small_075.in1k` |
| `MobileNetV3Small100` | 67.4 | 67.9 | 2.5 | `tf_mobilenetv3_small_100.in1k` |
| `MobileNetV3LargeMinimal100` | 72.3 | 72.3 | 3.9 | `tf_mobilenetv3_large_minimal_100.in1k` |
| `MobileNetV3Large075` | 73.3 | 73.4 | 4.0 | `tf_mobilenetv3_large_075.in1k` |
| `MobileNetV3Large100` | 75.2 | 77.9 | 5.5 | `mobilenetv3_large_100.miil_in21k_ft_in1k` |

### MobileViT &mdash; [paper](https://arxiv.org/abs/2110.02178)

Paper baselines: Table 2 of Mehta & Rastegari 2021.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MobileViTXXS` | 69.0 | 68.9 | 1.3 | `mobilevit_xxs.cvnets_in1k` |
| `MobileViTXS` | 74.8 | 74.6 | 2.3 | `mobilevit_xs.cvnets_in1k` |
| `MobileViTS` | 78.4 | 78.3 | 5.6 | `mobilevit_s.cvnets_in1k` |

### MobileViTV2 &mdash; [paper](https://arxiv.org/abs/2206.02680)

Paper baselines: Table 3 of Mehta & Rastegari 2022 (in1k from scratch, 256).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `MobileViTV2M050` | 70.2 | 70.2 | 1.4 | `mobilevitv2_050.cvnets_in1k` |
| `MobileViTV2M075` | 75.6 | 75.6 | 2.9 | `mobilevitv2_075.cvnets_in1k` |
| `MobileViTV2M100` | 78.1 | 78.1 | 4.9 | `mobilevitv2_100.cvnets_in1k` |
| `MobileViTV2M125` | 79.7 | 79.7 | 7.5 | `mobilevitv2_125.cvnets_in1k` |
| `MobileViTV2M150` | 80.4 | 82.6 | 11 | `mobilevitv2_150.cvnets_in22k_ft_in1k_384` |
| `MobileViTV2M175` | 80.8 | 82.9 | 14 | `mobilevitv2_175.cvnets_in22k_ft_in1k_384` |
| `MobileViTV2M200` | 81.2 | 83.4 | 18 | `mobilevitv2_200.cvnets_in22k_ft_in1k_384` |

### NextViT &mdash; [paper](https://arxiv.org/abs/2207.05501)

Paper baselines: Table 4 of Li et al. 2022 (in1k from scratch, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `NextViTSmall` | 82.5 | 86.0 | 32 | `nextvit_small.bd_ssld_6m_in1k_384` |
| `NextViTBase` | 83.2 | 86.4 | 45 | `nextvit_base.bd_ssld_6m_in1k_384` |
| `NextViTLarge` | 83.6 | 86.5 | 58 | `nextvit_large.bd_ssld_6m_in1k_384` |

### PiT &mdash; [paper](https://arxiv.org/abs/2103.16302)

Paper baselines: Table 4 of Heo et al. 2021 (in1k from scratch, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `PiT_Ti` | 73.0 | 72.9 | 4.8 | `pit_ti_224.in1k` |
| `PiT_Ti_Distilled` | 74.6 | 74.3 | 5.1 | `pit_ti_distilled_224.in1k` |
| `PiT_XS` | 78.1 | 78.2 | 11 | `pit_xs_224.in1k` |
| `PiT_XS_Distilled` | 79.1 | 79.2 | 11 | `pit_xs_distilled_224.in1k` |
| `PiT_S` | 80.9 | 81.1 | 23 | `pit_s_224.in1k` |
| `PiT_S_Distilled` | 81.9 | 81.8 | 24 | `pit_s_distilled_224.in1k` |
| `PiT_B` | 82.0 | 82.5 | 74 | `pit_b_224.in1k` |
| `PiT_B_Distilled` | 84.0 | 83.8 | 75 | `pit_b_distilled_224.in1k` |

### PoolFormer &mdash; [paper](https://arxiv.org/abs/2111.11418)

Paper baselines: Table 1 of Yu et al. 2021 (in1k from scratch, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `PoolFormerS12` | 77.2 | 77.2 | 12 | `poolformer_s12.sail_in1k` |
| `PoolFormerS24` | 80.3 | 80.3 | 21 | `poolformer_s24.sail_in1k` |
| `PoolFormerS36` | 81.4 | 81.4 | 31 | `poolformer_s36.sail_in1k` |
| `PoolFormerM36` | 82.1 | 82.1 | 56 | `poolformer_m36.sail_in1k` |
| `PoolFormerM48` | 82.5 | 82.5 | 73 | `poolformer_m48.sail_in1k` |

### Res2Net &mdash; [paper](https://arxiv.org/abs/1904.01169)

Paper baselines: Table 4 of Gao et al. 2019 (single-crop, 224).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `Res2Next50` | 78.2 | 78.2 | 25 | `res2next50.in1k` |
| `Res2Net50_14w_8s` | 78.0 | 78.1 | 25 | `res2net50_14w_8s.in1k` |
| `Res2Net50_48w_2s` | 77.5 | 77.5 | 25 | `res2net50_48w_2s.in1k` |
| `Res2Net50_26w_4s` | 78.0 | 78.0 | 26 | `res2net50_26w_4s.in1k` |
| `Res2Net50_26w_6s` | 78.7 | 78.6 | 37 | `res2net50_26w_6s.in1k` |
| `Res2Net101_26w_4s` | 79.2 | 79.2 | 45 | `res2net101_26w_4s.in1k` |
| `Res2Net50_26w_8s` | 79.2 | 79.0 | 48 | `res2net50_26w_8s.in1k` |

### ResMLP &mdash; [paper](https://arxiv.org/abs/2105.03404)

Paper baselines: Table 1 of Touvron et al. 2021 (no distillation).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ResMLP12` | 76.6 | 77.9 | 15 | `resmlp_12_224.fb_distilled_in1k` |
| `ResMLP24` | 79.4 | 80.8 | 30 | `resmlp_24_224.fb_distilled_in1k` |
| `ResMLP36` | 79.7 | 81.2 | 45 | `resmlp_36_224.fb_distilled_in1k` |
| `ResMLPBig24` | 81.0 | 84.4 | 129 | `resmlp_big_24_224.fb_in22k_ft_in1k` |

### ResNet &mdash; [paper](https://arxiv.org/abs/1512.03385)

Paper baselines: Table 5 of He et al. 2015 (single-crop, single-model on ImageNet val).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ResNet50` | 75.3 | 81.2 | 26 | `resnet50.a1_in1k` |
| `ResNet101` | 76.4 | 82.8 | 45 | `resnet101.a1h_in1k` |
| `ResNet152` | 77.0 | 83.5 | 60 | `resnet152.a1h_in1k` |

### ResNetV2 / BiT &mdash; [paper](https://arxiv.org/abs/1912.11370)

Paper baselines: Table 4 of Kolesnikov et al. 2019 (BiT-M = ImageNet-21k pretrained → in1k fine-tuned). The 152x2 variant is not in the paper main table.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ResNetV2_50x1` | 80.0 | 82.8 | 26 | `resnetv2_50x1_bit.goog_distilled_in1k` |
| `ResNetV2_101x1` | 82.5 | 82.3 | 45 | `resnetv2_101x1_bit.goog_in21k_ft_in1k` |
| `ResNetV2_50x3` | 84.0 | 84.0 | 217 | `resnetv2_50x3_bit.goog_in21k_ft_in1k` |
| `ResNetV2_152x2` | — | 84.5 | 236 | `resnetv2_152x2_bit.goog_in21k_ft_in1k` |
| `ResNetV2_101x3` | 84.7 | 84.4 | 388 | `resnetv2_101x3_bit.goog_in21k_ft_in1k` |
| `ResNetV2_152x4` | 85.4 | 84.9 | 937 | `resnetv2_152x4_bit.goog_in21k_ft_in1k` |

### ResNeXt &mdash; [paper](https://arxiv.org/abs/1611.05431)

Paper baselines: Table 3 of Xie et al. 2017 (single-crop, 224). The `32x16d` and `32x32d` Instagram-pretrained variants come from [Mahajan et al. 2018](https://arxiv.org/abs/1805.00932).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ResNeXt50_32x4d` | 77.8 | 82.2 | 25 | `resnext50_32x4d.fb_swsl_ig1b_ft_in1k` |
| `ResNeXt101_32x4d` | 78.8 | 83.3 | 44 | `resnext101_32x4d.fb_swsl_ig1b_ft_in1k` |
| `ResNeXt101_32x8d` | — | 84.3 | 89 | `resnext101_32x8d.fb_swsl_ig1b_ft_in1k` |
| `ResNeXt101_32x16d` | 84.2 | 84.2 | 194 | `resnext101_32x16d.fb_wsl_ig1b_ft_in1k` |
| `ResNeXt101_32x32d` | 85.1 | 85.1 | 469 | `resnext101_32x32d.fb_wsl_ig1b_ft_in1k` |

### SENet &mdash; [paper](https://arxiv.org/abs/1709.01507)

Paper baselines: Table 2 of Hu et al. 2017 (single-crop on ImageNet val).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `SEResNet50` | 77.6 | 81.3 | 28 | `seresnet50.ra2_in1k` |
| `SEResNeXt50_32x4d` | 79.1 | 82.2 | 28 | `seresnext50_32x4d.racm_in1k` |
| `SEResNeXt101_32x4d` | 80.2 | 80.9 | 49 | `seresnext101_32x4d.gluon_in1k` |
| `SEResNeXt101_32x8d` | — | 84.2 | 94 | `seresnext101_32x8d.ah_in1k` |

### Swin Transformer &mdash; [paper](https://arxiv.org/abs/2103.14030)

Paper baselines: Table 1 of Liu et al. 2021 (in1k from scratch, 224 unless noted).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `SwinTinyP4W7` | 81.3 | 81.4 | 28 | `swin_tiny_patch4_window7_224.ms_in1k` |
| `SwinSmallP4W7` | 83.0 | 83.3 | 50 | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` |
| `SwinBaseP4W7` | 83.5 | 85.3 | 88 | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| `SwinBaseP4W12` | 84.5 | 86.4 | 88 | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` |
| `SwinLargeP4W7` | 86.3 | 86.3 | 197 | `swin_large_patch4_window7_224.ms_in22k_ft_in1k` |
| `SwinLargeP4W12` | 87.3 | 87.1 | 197 | `swin_large_patch4_window12_384.ms_in22k_ft_in1k` |

### Swin Transformer V2 &mdash; [paper](https://arxiv.org/abs/2111.09883)

Paper baselines: Table 1 of Liu et al. 2021 (in1k from scratch, single-crop). Multiple window-size variants come from timm.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `SwinV2TinyW8` | 81.8 | 81.8 | 28 | `swinv2_tiny_window8_256.ms_in1k` |
| `SwinV2TinyW16` | — | 82.8 | 28 | `swinv2_tiny_window16_256.ms_in1k` |
| `SwinV2SmallW8` | 83.7 | 83.8 | 50 | `swinv2_small_window8_256.ms_in1k` |
| `SwinV2SmallW16` | — | 84.2 | 50 | `swinv2_small_window16_256.ms_in1k` |
| `SwinV2BaseW8` | 84.2 | 84.2 | 88 | `swinv2_base_window8_256.ms_in1k` |
| `SwinV2BaseW16` | — | 84.6 | 88 | `swinv2_base_window16_256.ms_in1k` |
| `SwinV2BaseW12` | 87.1 | 87.1 | 88 | `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` |
| `SwinV2LargeW12` | 87.6 | 87.5 | 197 | `swinv2_large_window12to24_192to384.ms_in22k_ft_in1k` |

### VGG &mdash; [paper](https://arxiv.org/abs/1409.1556)

Paper baselines: Table 4 of Simonyan & Zisserman 2014 (single-crop, single-model). BN variants are not in the original paper (added by torchvision).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `VGG16` | 71.5 | 71.6 | 138 | `vgg16.tv_in1k` |
| `VGG16_BN` | — | 73.4 | 138 | `vgg16_bn.tv_in1k` |
| `VGG19` | 71.3 | 72.4 | 144 | `vgg19.tv_in1k` |
| `VGG19_BN` | — | 74.2 | 144 | `vgg19_bn.tv_in1k` |

### ViT &mdash; [paper](https://arxiv.org/abs/2010.11929)

Paper baselines: Table 5 of Dosovitskiy et al. 2020 (ImageNet-21k pretrained → in1k fine-tuned at 384). The Tiny/Small variants come from the [augreg paper](https://arxiv.org/abs/2106.10270).

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `ViTTiny16` | 75.5 | 78.4 | 5.8 | `vit_tiny_patch16_384.augreg_in21k_ft_in1k` |
| `ViTSmall32` | 77.0 | 80.5 | 23 | `vit_small_patch32_384.augreg_in21k_ft_in1k` |
| `ViTSmall16` | 81.4 | 83.8 | 22 | `vit_small_patch16_384.augreg_in21k_ft_in1k` |
| `ViTBase32` | 81.3 | 83.4 | 88 | `vit_base_patch32_384.augreg_in21k_ft_in1k` |
| `ViTBase16` | 84.2 | 86.0 | 87 | `vit_base_patch16_384.augreg_in21k_ft_in1k` |
| `ViTLarge32` | 81.5 | 81.5 | 307 | `vit_large_patch32_384.orig_in21k_ft_in1k` |
| `ViTLarge16` | 85.2 | 87.1 | 305 | `vit_large_patch16_384.augreg_in21k_ft_in1k` |

### Xception &mdash; [paper](https://arxiv.org/abs/1610.02357)

Paper baseline: Table 1 of Chollet 2017 (single-crop on ImageNet val). kmodels uses Xception-65; the original paper reports the Xception-71 architecture with slightly different numbers.

| Variant | Paper Top-1 (%) | Best Top-1 (%) | Params (M) | Source (best) |
|---------|----------------:|---------------:|-----------:|---------------|
| `Xception` | 79.0 | 83.2 | 40 | `xception65.ra3_in1k` |
