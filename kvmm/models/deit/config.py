DEIT_MODEL_CONFIG = {
    "DEiTTiny16": {
        "patch_size": 16,
        "dim": 192,
        "depth": 12,
        "num_heads": 3,
        "use_distillation": False,
    },
    "DEiTSmall16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "use_distillation": False,
    },
    "DEiTBase16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "use_distillation": False,
    },
    "DEiTTinyDistilled16": {
        "patch_size": 16,
        "dim": 192,
        "depth": 12,
        "num_heads": 3,
        "use_distillation": True,
    },
    "DEiTSmallDistilled16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "use_distillation": True,
    },
    "DEiTBaseDistilled16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "use_distillation": True,
    },
    "DEiT3Small16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "no_embed_class": True,
        "init_values": 1e-6,
    },
    "DEiT3Medium16": {
        "patch_size": 16,
        "dim": 512,
        "depth": 12,
        "num_heads": 8,
        "no_embed_class": True,
        "init_values": 1e-6,
    },
    "DEiT3Base16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "init_values": 1e-6,
    },
    "DEiT3Large16": {
        "patch_size": 16,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "no_embed_class": True,
        "init_values": 1e-6,
    },
    "DEiT3Huge14": {
        "patch_size": 14,
        "dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "no_embed_class": True,
        "init_values": 1e-6,
    },
}

DEIT_WEIGHTS_CONFIG = {
    "DEiTTiny16": {
        "fb_in1k_224": {
            "url": "",
        },
    },
    "DEiTSmall16": {
        "fb_in1k_224": {
            "url": "",
        },
    },
    "DEiTBase16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in1k_384": {
            "url": "",
        },
    },
    "DEiTTinyDistilled16": {
        "fb_in1k_224": {
            "url": "",
        },
    },
    "DEiTSmallDistilled16": {
        "fb_in1k_224": {
            "url": "",
        },
    },
    "DEiTBaseDistilled16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in1k_384": {
            "url": "",
        },
    },
    "DEiT3Small16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in1k_384": {
            "url": "",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "",
        },
    },
    "DEiT3Medium16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "",
        },
    },
    "DEiT3Base16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in1k_384": {
            "url": "",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "",
        },
    },
    "DEiT3Large16": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in1k_384": {
            "url": "",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "",
        },
    },
    "DEiT3Huge14": {
        "fb_in1k_224": {
            "url": "",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "",
        },
    },
}
