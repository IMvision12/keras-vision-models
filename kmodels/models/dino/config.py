DINO_VIT_MODEL_CONFIG = {
    "DinoViTSmall16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
    },
    "DinoViTSmall8": {
        "patch_size": 8,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
    },
    "DinoViTBase16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
    },
    "DinoViTBase8": {
        "patch_size": 8,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
    },
}

DINO_RESNET_MODEL_CONFIG = {
    "DinoResNet50": {
        "block_fn": "bottleneck_block",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
}

DINO_WEIGHTS_CONFIG = {
    "DinoViTSmall16": {
        "dino": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino/dino_vits16.weights.h5",
        },
    },
    "DinoViTSmall8": {
        "dino": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino/dino_vits8.weights.h5",
        },
    },
    "DinoViTBase16": {
        "dino": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino/dino_vitb16.weights.h5",
        },
    },
    "DinoViTBase8": {
        "dino": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino/dino_vitb8.weights.h5",
        },
    },
    "DinoResNet50": {
        "dino": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino/dino_resnet50.weights.h5",
        },
    },
}
