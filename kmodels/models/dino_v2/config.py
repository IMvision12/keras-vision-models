DINOV2_MODEL_CONFIG = {
    "DinoV2Small14": {
        "patch_size": 14,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "init_values": 1.0,
    },
    "DinoV2Base14": {
        "patch_size": 14,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "init_values": 1.0,
    },
    "DinoV2Large14": {
        "patch_size": 14,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "init_values": 1.0,
    },
}

DINOV2_WEIGHTS_CONFIG = {
    "DinoV2Small14": {
        "dinov2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v2/dinov2_vits14.weights.h5",
        },
    },
    "DinoV2Base14": {
        "dinov2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v2/dinov2_vitb14.weights.h5",
        },
    },
    "DinoV2Large14": {
        "dinov2": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/dino_v2/dinov2_vitl14.weights.h5",
        },
    },
}
