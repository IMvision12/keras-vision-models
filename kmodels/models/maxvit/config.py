MAXVIT_MODEL_CONFIG = {
    "MaxViTTiny": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
    },
    "MaxViTSmall": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
    },
    "MaxViTBase": {
        "stem_width": 64,
        "depths": [2, 6, 14, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
    },
    "MaxViTLarge": {
        "stem_width": 128,
        "depths": [2, 6, 14, 2],
        "embed_dim": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
    },
    "MaxViTXLarge": {
        "stem_width": 192,
        "depths": [2, 6, 14, 2],
        "embed_dim": [192, 384, 768, 1536],
        "num_heads": [6, 12, 24, 48],
    },
}

MAXVIT_WEIGHTS_CONFIG = {
    "MaxViTTiny": {
        "in1k_224": {"url": ""},
        "in1k_384": {"url": ""},
        "in1k_512": {"url": ""},
    },
    "MaxViTSmall": {
        "in1k_224": {"url": ""},
        "in1k_384": {"url": ""},
        "in1k_512": {"url": ""},
    },
    "MaxViTBase": {
        "in1k_224": {"url": ""},
        "in1k_384": {"url": ""},
        "in1k_512": {"url": ""},
        "in21k_224": {"url": ""},
        "in21k_ft_in1k_384": {"url": ""},
        "in21k_ft_in1k_512": {"url": ""},
    },
    "MaxViTLarge": {
        "in1k_224": {"url": ""},
        "in1k_384": {"url": ""},
        "in1k_512": {"url": ""},
        "in21k_224": {"url": ""},
        "in21k_ft_in1k_384": {"url": ""},
        "in21k_ft_in1k_512": {"url": ""},
    },
    "MaxViTXLarge": {
        "in21k_224": {"url": ""},
        "in21k_ft_in1k_384": {"url": ""},
        "in21k_ft_in1k_512": {"url": ""},
    },
}
