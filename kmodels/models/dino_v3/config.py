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
