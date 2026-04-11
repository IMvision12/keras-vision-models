# Object Detection — Per-Variant COCO AP & Params

COCO val2017 box AP and parameter counts for **every object detection variant in kmodels**, taken from the **original publication's main results table** for that variant. `Params (M)` is computed by directly instantiating the model from the kmodels registry. Variants are listed in increasing parameter order within each family.

The metric is COCO val2017 single-scale **box AP** (not multi-scale TTA). Each family section cites the exact paper Table the values came from in its intro paragraph. `—` means the paper doesn't report that variant or the value is not reliably documented.

---

### DETR &mdash; [paper](https://arxiv.org/abs/2005.12872)


| Variant | Box AP | AP50 | AP75 | Params (M) |
|---------|------:|----:|----:|-----------:|
| `DETRResNet50`  | 42.0 | 62.4 | 44.2 | 41 |
| `DETRResNet101` | 43.5 | 63.8 | 46.4 | 60 |

### RT-DETR &mdash; [paper](https://arxiv.org/abs/2304.08069)


| Variant | Weights | Box AP | Params (M) |
|---------|---------|------:|-----------:|
| `RTDETRResNet18`  | `coco`      | 46.5 | 20 |
| `RTDETRResNet18`  | `coco_o365` | 49.2 | 20 |
| `RTDETRResNet34`  | `coco`      | 48.9 | 31 |
| `RTDETRResNet50`  | `coco`      | 53.1 | 43 |
| `RTDETRResNet50`  | `coco_o365` | 55.3 | 43 |
| `RTDETRResNet101` | `coco`      | 54.3 | 76 |
| `RTDETRResNet101` | `coco_o365` | 56.2 | 76 |

### RT-DETRv2 &mdash; [paper](https://arxiv.org/abs/2407.17140)


| Variant | Box AP | Params (M) |
|---------|------:|-----------:|
| `RTDETRV2ResNet18`  | 47.9 | 20 |
| `RTDETRV2ResNet34`  | 49.9 | 31 |
| `RTDETRV2ResNet50`  | 53.4 | 43 |
| `RTDETRV2ResNet101` | 54.3 | 76 |

### D-FINE &mdash; [paper](https://arxiv.org/abs/2410.13842)


| Variant | Weights | Box AP | Params (M) |
|---------|---------|------:|-----------:|
| `DFineNano`   | COCO        | 42.8 | 3.8 |
| `DFineNano`   | COCO+O365   | 44.2 | 3.8 |
| `DFineSmall`  | COCO        | 48.7 | 10  |
| `DFineSmall`  | COCO+O365   | 50.7 | 10  |
| `DFineMedium` | COCO        | 52.3 | 19  |
| `DFineMedium` | COCO+O365   | 55.1 | 19  |
| `DFineLarge`  | COCO        | 54.0 | 31  |
| `DFineLarge`  | COCO+O365   | 57.1 | 31  |
| `DFineXLarge` | COCO        | 55.8 | 63  |
| `DFineXLarge` | COCO+O365   | 59.3 | 63  |

### RF-DETR &mdash; [paper](https://arxiv.org/abs/2511.09554)


| Variant | Box AP | AP50 | AP75 | Params (M) |
|---------|------:|----:|----:|-----------:|
| `RFDETRNano`   | 48.0 | 67.0 | 51.4 | 27 |
| `RFDETRSmall`  | 52.9 | 71.9 | 57.0 | 28 |
| `RFDETRMedium` | 54.7 | 73.5 | 59.2 | 30 |
| `RFDETRBase`   | —    | —    | —    | 27 |
| `RFDETRLarge`  | 56.5 | 75.1 | 61.3 | 30 |
