VIT_MODEL_CONFIG = {
    "vit_tiny_patch16": {
        "patch_size": 16,
        "dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_small_patch16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_small_patch32": {
        "patch_size": 32,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_base_patch16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_base_patch32": {
        "patch_size": 32,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_large_patch16": {
        "patch_size": 16,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
    "vit_large_patch32": {
        "patch_size": 32,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
    },
}

VIT_WEIGHTS_CONFIG = {
    "ViTTiny16": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_tiny_patch16_384_augreg_in21k_ft_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_tiny_patch16_224_augreg_in21k.keras",
        },
    },
    "ViTSmall16": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_small_patch16_384_augreg_in21k_ft_in1k.keras",
        },
        "augreg_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_small_patch16_384_augreg_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_small_patch16_224_augreg_in21k.keras",
        },
    },
    "ViTSmall32": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_small_patch32_384_augreg_in21k_ft_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_small_patch32_224_augreg_in21k.keras",
        },
    },
    "ViTBase16": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch16_384_augreg_in21k_ft_in1k.keras",
        },
        "orig_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch16_384_orig_in21k_ft_in1k.keras",
        },
        "augreg_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch16_384_augreg_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch16_224_augreg_in21k.keras",
        },
    },
    "ViTBase32": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch32_384_augreg_in21k_ft_in1k.keras",
        },
        "augreg_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch32_384_augreg_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_base_patch32_224_augreg_in21k.keras",
        },
    },
    "ViTLarge16": {
        "augreg_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_large_patch16_384_augreg_in21k_ft_in1k.keras",
        },
        "augreg_in21k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_large_patch16_224_augreg_in21k.keras",
        },
    },
    "ViTLarge32": {
        "orig_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.2/vit_large_patch32_384_orig_in21k_ft_in1k.keras",
        },
    },
}
