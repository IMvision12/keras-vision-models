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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_tiny_patch16_224_fb_in1k.weights.h5",
        },
    },
    "DEiTSmall16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_small_patch16_224_fb_in1k.weights.h5",
        },
    },
    "DEiTBase16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_base_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_base_patch16_384_fb_in1k.weights.h5",
        },
    },
    "DEiTTinyDistilled16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_tiny_distilled_patch16_224_fb_in1k.weights.h5",
        },
    },
    "DEiTSmallDistilled16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_small_distilled_patch16_224_fb_in1k.weights.h5",
        },
    },
    "DEiTBaseDistilled16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_base_distilled_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit_base_distilled_patch16_384_fb_in1k.weights.h5",
        },
    },
    "DEiT3Small16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_small_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_small_patch16_384_fb_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_small_patch16_224_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_small_patch16_384_fb_in22k_ft_in1k.weights.h5",
        },
    },
    "DEiT3Medium16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_medium_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_medium_patch16_224_fb_in22k_ft_in1k.weights.h5",
        },
    },
    "DEiT3Base16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_base_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_base_patch16_384_fb_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_base_patch16_224_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_base_patch16_384_fb_in22k_ft_in1k.weights.h5",
        },
    },
    "DEiT3Large16": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_large_patch16_224_fb_in1k.weights.h5",
        },
        "fb_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_large_patch16_384_fb_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_large_patch16_224_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_large_patch16_384_fb_in22k_ft_in1k.weights.h5",
        },
    },
    "DEiT3Huge14": {
        "fb_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_huge_patch14_224_fb_in1k.weights.json",
        },
        "fb_in22k_ft_in1k_224": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/deit3_huge_patch14_224_fb_in22k_ft_in1k.weights.json",
        },
    },
}
