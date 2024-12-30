CONVNEXT_MODEL_CONFIG = {
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
    "small": {
        "depths": [3, 3, 27, 3],
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
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
    },
}

CONVNEXT_WEIGHTS_CONFIG = {
    # Timm specific variants
    "ConvNeXtAtto": {
        "d2_in1k": {
            "url": "",
        },
    },
    "ConvNeXtFemto": {
        "d1_in1k": {
            "url": "",
        }
    },
    "ConvNeXtPico": {
        "d1_in1k": {
            "url": "",
        }
    },
    "ConvNeXtNano": {
        "d1h_in1k": {
            "url": "",
        },
        "in12k_ft_in1k": {
            "url": "",
        },
    },
    # ConvNeXtV1
    "ConvNeXtTiny": {
        "fb_in1k": {
            "url": "",
        },
        "fb_in22k": {
            "url": "",
        },
        "fb_in22k_ft_in1k": {
            "url": "",
        },
    },
    "ConvNeXtSmall": {
        "fb_in1k": {
            "url": "",
        },
        "fb_in22k": {
            "url": "",
        },
        "fb_in22k_ft_in1k": {
            "url": "",
        },
    },
    "ConvNeXtBase": {
        "fb_in1k": {
            "url": "",
        },
        "fb_in22k": {
            "url": "",
        },
        "fb_in22k_ft_in1k": {
            "url": "",
        },
    },
    "ConvNeXtLarge": {
        "fb_in1k": {
            "url": "",
        },
        "fb_in22k": {
            "url": "",
        },
        "fb_in22k_ft_in1k": {
            "url": "",
        },
    },
    "ConvNeXtXLarge": {
        "fb_in22k": {
            "url": "",
        },
        "fb_in22k_ft_in1k": {
            "url": "",
        },
    },
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
    },
    "ConvNeXtV2Tiny": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
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
    },
    "ConvNeXtV2Large": {
        "fcmae_ft_in1k": {
            "url": "",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "",
        },
    },
}
