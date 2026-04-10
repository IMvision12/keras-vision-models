DINOV3_VIT_MODEL_CONFIG = {
    "DinoV3ViTSmall16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "init_values": 1.0,
        "rope_theta": 100.0,
    },
    "DinoV3ViTBase16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "init_values": 1.0,
        "rope_theta": 100.0,
    },
    "DinoV3ViTLarge16": {
        "patch_size": 16,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "init_values": 1.0,
        "rope_theta": 100.0,
    },
}

DINOV3_CONVNEXT_MODEL_CONFIG = {
    "DinoV3ConvNeXtTiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "DinoV3ConvNeXtSmall": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "DinoV3ConvNeXtBase": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "DinoV3ConvNeXtLarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
}

# HuggingFace model IDs (for conversion script reference).
DINOV3_HF_MODEL_IDS = {
    "DinoV3ViTSmall16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "DinoV3ViTBase16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "DinoV3ViTLarge16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "DinoV3ConvNeXtTiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "DinoV3ConvNeXtSmall": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "DinoV3ConvNeXtBase": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "DinoV3ConvNeXtLarge": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

# Placeholder weight URLs (update after running conversion script and uploading).
DINOV3_WEIGHTS_CONFIG = {
    "DinoV3ViTSmall16": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_vits16.weights.h5",
        },
    },
    "DinoV3ViTBase16": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_vitb16.weights.h5",
        },
    },
    "DinoV3ViTLarge16": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_vitl16.weights.h5",
        },
    },
    "DinoV3ConvNeXtTiny": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_convnext_tiny.weights.h5",
        },
    },
    "DinoV3ConvNeXtSmall": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_convnext_small.weights.h5",
        },
    },
    "DinoV3ConvNeXtBase": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_convnext_base.weights.h5",
        },
    },
    "DinoV3ConvNeXtLarge": {
        "dinov3": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v3/dinov3_convnext_large.weights.h5",
        },
    },
}
