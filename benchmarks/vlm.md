# Vision-Language Models — Params

Parameter counts for every vision-language model family in kmodels. Values are reported as **min–max ranges across all variants** in each family (image encoder + text encoder combined), computed by directly instantiating the smallest and largest variant from the kmodels registry.

| 🏷️ Family | 🔢 Params (M) | 🧩 Variants | 📦 Weights |
|----------|--------------:|------------|-----------|
| CLIP    | 151–2500 | ViT-B/16, ViT-B/32, ViT-L/14, ViT-G/14, ViT-BigG/14 | `transformers` |
| SigLIP  | 203–877  | Base/16, Large/16, So400m/14 | `transformers` |
| SigLIP2 | 377–1100 | Base/32, Base/16, Large/16, So400m/16, So400m/14 | `transformers` |

## Picking a Model

| Constraint | Good starting points |
|------------|----------------------|
| **Smallest zero-shot classifier** (< 200M) | CLIP ViT-B/16, CLIP ViT-B/32 |
| **Solid mid-range zero-shot** (200–500M) | SigLIP Base/16, SigLIP2 Base/16 |
| **Best zero-shot quality** | CLIP BigG/14, SigLIP So400m/14, SigLIP2 So400m/14 |
| **Multilingual / dense features** | SigLIP2 (256K vocab, improved localization) |
| **Pure contrastive InfoNCE** | CLIP |
| **Sigmoid contrastive (better at small batch)** | SigLIP, SigLIP2 |
