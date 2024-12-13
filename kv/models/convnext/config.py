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
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_atto_d2_in1k.keras",
        },
    },
    "ConvNeXtFemto": {
        "d1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_femto_d1_in1k.keras",
        }
    },
    "ConvNeXtPico": {
        "d1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_pico_d1_in1k.keras",
        }
    },
    "ConvNeXtNano": {
        "d1h_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_nano_d1h_in1k.keras",
        },
        "in12k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_nano_in12k_ft_in1k.keras",
        },
    },
    # ConvNeXtV1
    "ConvNeXtTiny": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_tiny_fb_in1k.keras",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_tiny_fb_in22k.keras",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_tiny_fb_in22k_ft_in1k.keras",
        },
    },
    "ConvNeXtSmall": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_small_fb_in1k.keras",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_small_fb_in22k.keras",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_small_fb_in22k_ft_in1k.keras",
        },
    },
    "ConvNeXtBase": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_base_fb_in1k.keras",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_base_fb_in22k.keras",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_base_fb_in22k_ft_in1k.keras",
        },
    },
    "ConvNeXtLarge": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_large_fb_in1k.keras",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_large_fb_in22k.keras",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_large_fb_in22k_ft_in1k.keras",
        },
    },
    "ConvNeXtXLarge": {
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_xlarge_fb_in22k.keras",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnext_xlarge_fb_in22k_ft_in1k.keras",
        },
    },
    # ConvNeXtV2
    "ConvNeXtV2Atto": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_atto_fcmae_ft_in1k.keras",
        }
    },
    "ConvNeXtV2Femto": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_femto_fcmae_ft_in1k.keras",
        }
    },
    "ConvNeXtV2Pico": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_pico_fcmae_ft_in1k.keras",
        }
    },
    "ConvNeXtV2Nano": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_nano_fcmae_ft_in1k.keras",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_nano_fcmae_ft_in22k_in1k.keras",
        },
    },
    "ConvNeXtV2Tiny": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in1k.keras",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in22k_in1k.keras",
        },
    },
    "ConvNeXtV2Base": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_base_fcmae_ft_in1k.keras",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_base_fcmae_ft_in22k_in1k.keras",
        },
    },
    "ConvNeXtV2Large": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_large_fcmae_ft_in1k.keras",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convnextv2_large_fcmae_ft_in22k_in1k.keras",
        },
    },
}
