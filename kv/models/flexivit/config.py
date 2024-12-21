FLEXIVIT_MODEL_CONFIG = {
    "flexivit_small": {
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
    "flexivit_base": {
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
    "flexivit_large": {
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
            "url": "",
        },
        "600ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "",
        },
    },
    "FlexiViTBase": {
        "1200ep_in1k": {
            "url": "",
        },
        "600ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "",
        },
        "1000ep_in21k": {
            "url": "",
        },
        "300ep_in21k": {
            "url": "",
        },
    },
    "FlexiViTLarge": {
        "1200ep_in1k": {
            "url": "",
        },
        "600ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "",
        },
        "300ep_in1k": {
            "url": "",
        },
    },
}
