# Segmentation — Per-Variant Paper Metrics & Params

Paper-reported segmentation metrics and parameter counts for **every segmentation variant in kmodels**, taken from the **original publication's main results table** for that variant. `Params (M)` is computed by directly instantiating the model from the kmodels registry. Variants are listed in increasing parameter order within each family.

The headline metric varies by family because each task uses a different evaluation protocol:

- **Semantic segmentation** (DeepLabV3, SegFormer, EoMT-semantic): mean IoU (mIoU) on the relevant validation set
- **Panoptic segmentation** (EoMT-panoptic): Panoptic Quality (PQ)
- **Promptable segmentation** (SAM, SAM 2): primary metric is zero-shot IoU averaged across many datasets and click-prompt budgets — there is no single canonical "headline number" comparable across variants, so the numbers below are from the paper's main aggregate Table where available
- **Open-vocabulary detection + segmentation** (SAM 3): box/mask AP on LVIS

Each family section cites the exact paper Table the values came from. `—` means the paper doesn't report that variant or the value is not reliably documented in a single canonical table.

---

### DeepLabV3 &mdash; [paper](https://arxiv.org/abs/1706.05587)


| Variant | Paper mIoU (%) | Dataset | Params (M) | Source |
|---------|---------------:|---------|-----------:|--------|
| `DeepLabV3ResNet50`  | — | — | 40 | torchvision-trained, paper does not break out R50 vs R101 in the main result |
| `DeepLabV3ResNet101` | 79.3 | PASCAL VOC 2012 val | 59 | DeepLabV3 Table 4 (R101 baseline, no JFT) |

### EoMT &mdash; [paper](https://arxiv.org/abs/2503.19108)


| Variant | PQ (no mask) | PQ (w/ mask) | Params (M) |
|---------|------------:|------------:|-----------:|
| `EoMTSmall` | 44.7 | 46.1 | 24  |
| `EoMTBase`  | 50.6 | 51.5 | 93  |
| `EoMTLarge` | 56.0 | 56.2 | 316 |

For the ViT-L variant only, the paper provides additional task results (Tables 4–6) showing how the same architecture transfers across protocols:

| Task / Dataset | Input | Metric | Value | Source |
|----------------|-------|--------|------:|--------|
| COCO panoptic | 640² | PQ | 56.0 | Table 4 |
| COCO panoptic | 1280² | PQ | 58.3 | Table 4 |
| COCO instance | 640² | mask AP | 45.2 | Table 6 |
| COCO instance | 1280² | mask AP | 48.8 | Table 6 |
| ADE20K panoptic† | 640² | PQ | 50.6 | Table 4 |
| ADE20K panoptic† | 1280² | PQ | 51.7 | Table 4 |
| Cityscapes semantic | 1024² | mIoU | 84.2 | Table 5 |
| ADE20K semantic | 512² | mIoU | 58.4 | Table 5 |

†Pre-trained for COCO panoptic before transfer.

EoMT also matches ViT-Adapter + Mask2Former on quality while running **up to ~4× faster** thanks to its plain-ViT design (e.g. ViT-L: 128 FPS vs. 29 FPS). The paper additionally evaluates a much larger ViT-g variant (1164M params, 57.0 PQ at 640² / 59.2 PQ at 1280²) which is not currently exposed in kmodels.

### SAM &mdash; [paper](https://arxiv.org/abs/2304.02643)


- **1-click** = single positive click, top-ranked of SAM's three mask outputs
- **1-click (oracle)** = single positive click, best of SAM's three mask outputs against the ground truth

Values are read from Figure 13 (right) with ~0.3 mIoU precision since the paper doesn't tabulate them. The paper's text observes that "ViT-H improves substantially over ViT-B, but has only marginal gains over ViT-L".

| Variant | 1-click mIoU (%) | 1-click oracle mIoU (%) | Params (M) |
|---------|-----------------:|------------------------:|-----------:|
| `SAMViTBase`  | 57.0 | 69.0 | 94  |
| `SAMViTLarge` | 58.5 | 71.5 | 308 |
| `SAMViTHuge`  | 59.0 | 72.5 | 641 |

The SAM 2 paper's Table 5 gives a tabulated counterpart for SAM (ViT-H) on a slightly extended evaluation: **58.1** 1-click mIoU on SA-23 (and **81.3** with 5 clicks), confirming that SAM ViT-H sits in the high-58/low-59 range. SAM also reports, for the default ViT-H release: zero-shot mask AR@1000 = **59.3** on LVIS object proposals (Table 4); edge-detection on BSDS500 = ODS **0.768** / R50 **0.928** (Table 3); and instance segmentation on LVIS with ViTDet boxes = mask AP **44.7** (Table 5).

### SAM 2 &mdash; [paper](https://arxiv.org/abs/2408.00714)


| Variant | SA-V val | SA-V test | MOSE val | DAVIS17 val | LVOS val | Params (M) |
|---------|---------:|----------:|---------:|------------:|---------:|-----------:|
| `Sam2Tiny`     | 75.2 | 76.5 | 71.8 | 89.4 | 77.5 | 38  |
| `Sam2Small`    | 77.0 | 76.6 | 73.5 | 89.6 | 77.3 | 46  |
| `Sam2BasePlus` | 77.5 | 78.2 | 73.8 | 90.0 | 77.7 | 81  |
| `Sam2Large`    | 78.6 | 79.5 | 74.6 | 90.2 | 80.1 | 226 |

All values are J&F mean (%). SAM 2 sets a new state of the art on every benchmark above by a wide margin — for example, on SA-V val, the prior-best Cutie-base+ achieves 61.3 J&F vs. SAM 2 (Hiera-T) at 75.2 (+13.9 with the smallest variant). On MOSE val, prior-best Cutie-base+ scores 71.7 vs. SAM 2 (Hiera-L) at 74.6.

### SAM 3 &mdash; [paper](https://arxiv.org/abs/2511.16719)


| Variant | LVIS mask AP | LVIS box AP | COCO box AP | SA-Co/Gold cgF1 | Params (M) |
|---------|-------------:|------------:|-----------:|---------------:|-----------:|
| `SAM3` | 48.5 | 53.6 | 56.4 | 54.1 | 478 |

Additional metrics from Table 1: ADE-847 mIoU = 13.8, PascalContext-59 mIoU = 60.8, Cityscapes mIoU = 65.2, COCO-O AP_o = 55.7. SAM 3 sets a new state-of-the-art on closed-vocabulary COCO and on LVIS box/mask AP, and roughly **doubles** the prior best (OWLv2*) on the SA-Co/Gold open-vocabulary benchmark.

### SegFormer &mdash; [paper](https://arxiv.org/abs/2105.15203)


| Variant | Cityscapes mIoU (%) | ADE20K mIoU (%) | Params (M) | Source |
|---------|--------------------:|----------------:|-----------:|--------|
| `SegFormerB0` | 76.2 | 37.4 | 3.8 | SegFormer Table 1 / 7 |
| `SegFormerB1` | 78.5 | 42.2 | 14  | SegFormer Table 1 / 7 |
| `SegFormerB2` | 81.0 | 46.5 | 28  | SegFormer Table 1 / 7 |
| `SegFormerB3` | 81.7 | 49.4 | 47  | SegFormer Table 1 / 7 |
| `SegFormerB4` | 82.3 | 50.3 | 64  | SegFormer Table 1 / 7 |
| `SegFormerB5` | 82.4 | 51.0 | 85  | SegFormer Table 1 / 7 |
