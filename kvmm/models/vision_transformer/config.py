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
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTSmall16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in1k_224": {
            "url": "",
        },
        "augreg_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTSmall32": {
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTBase16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "orig_in21k_ft_in1k_224": {
            "url": "",
        },
        "orig_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in1k_224": {
            "url": "",
        },
        "augreg_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTBase32": {
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in1k_224": {
            "url": "",
        },
        "augreg_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTLarge16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "",
        },
        "augreg_in21k_224": {
            "url": "",
        },
    },
    "ViTLarge32": {
        "orig_in21k_ft_in1k_384": {
            "url": "",
        },
    },
}
