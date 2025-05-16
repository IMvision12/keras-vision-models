VIT_MODEL_CONFIG = {
    "ViTTiny16": {
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
    "ViTSmall16": {
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
    "ViTSmall32": {
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
    "ViTBase16": {
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
    "ViTBase32": {
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
    "ViTLarge16": {
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
    "ViTLarge32": {
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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_tiny_patch16_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_tiny_patch16_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_tiny_patch16_224_augreg_in21k.weights.h5",
        },
    },
    "ViTSmall16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch16_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch16_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch16_224_augreg_in1k.weights.h5",
        },
        "augreg_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch16_384_augreg_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch16_224_augreg_in21k.weights.h5",
        },
    },
    "ViTSmall32": {
        "augreg_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch32_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch32_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_small_patch32_224_augreg_in21k.weights.h5",
        },
    },
    "ViTBase16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "orig_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_224_orig_in21k_ft_in1k.weights.h5",
        },
        "orig_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_384_orig_in21k_ft_in1k.weights.h5",
        },
        "augreg_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_224_augreg_in1k.weights.h5",
        },
        "augreg_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_384_augreg_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch16_224_augreg_in21k.weights.h5",
        },
    },
    "ViTBase32": {
        "augreg_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch32_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch32_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch32_224_augreg_in1k.weights.h5",
        },
        "augreg_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch32_384_augreg_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_base_patch32_224_augreg_in21k.weights.h5",
        },
    },
    "ViTLarge16": {
        "augreg_in21k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_large_patch16_224_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_large_patch16_384_augreg_in21k_ft_in1k.weights.h5",
        },
        "augreg_in21k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_large_patch16_224_augreg_in21k.weights.h5",
        },
    },
    "ViTLarge32": {
        "orig_in21k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/vit_large_patch32_384_orig_in21k_ft_in1k.weights.h5",
        },
    },
}
