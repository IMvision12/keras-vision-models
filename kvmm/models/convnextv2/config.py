CONVNEXTV2_MODEL_CONFIG = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [40, 80, 160, 320],
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [48, 96, 192, 384],
    },
    "pico": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [64, 128, 256, 512],
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "huge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [352, 704, 1408, 2816],
    },
}

CONVNEXTV2_WEIGHTS_CONFIG = {
    # ConvNeXtV2
    "ConvNeXtV2Atto": {
        "fcmae_ft_in1k": {
            "url": "",
        }
    },
    "ConvNeXtV2Femto": {
        "fcmae_ft_in1k": {
            "url": "",
        }
    },
    "ConvNeXtV2Pico": {
        "fcmae_ft_in1k": {
            "url": "",
        }
    },
    "ConvNeXtV2Nano": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "",
        },
    },
    "ConvNeXtV2Tiny": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "",
        },
    },
    "ConvNeXtV2Base": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "",
        },
    },
    "ConvNeXtV2Large": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "",
        },
    },
    "ConvNeXtV2Huge": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k_512": {
            "url": "",
        },
    },
}
