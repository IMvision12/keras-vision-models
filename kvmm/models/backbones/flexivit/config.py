FLEXIVIT_MODEL_CONFIG = {
    "FlexiViTSmall": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
        "no_embed_class": True,
    },
    "FlexiViTBase": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
        "no_embed_class": True,
    },
    "FlexiViTLarge": {
        "patch_size": 16,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.0,
        "no_embed_class": True,
    },
}

FLEXIVIT_WEIGHTS_CONFIG = {
    "FlexiViTSmall": {
        "1200ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_small_1200ep_in1k.weights.h5",
        },
        "600ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_small_600ep_in1k.weights.h5",
        },
        "300ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_small_300ep_in1k.weights.h5",
        },
    },
    "FlexiViTBase": {
        "1200ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_base_1200ep_in1k.weights.h5",
        },
        "600ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_base_300ep_in1k.weights.h5",
        },
        "1000ep_in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_base_1000ep_in21k.weights.h5",
        },
        "300ep_in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_base_300ep_in21k.weights.h5",
        },
    },
    "FlexiViTLarge": {
        "1200ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_large_1200ep_in1k.weights.h5",
        },
        "600ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_large_600ep_in1k.weights.h5",
        },
        "300ep_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/flexivit_large_300ep_in1k.weights.h5",
        },
    },
}
