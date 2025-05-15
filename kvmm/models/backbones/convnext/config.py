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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_atto_d2_in1k.weights.h5",
        },
    },
    "ConvNeXtFemto": {
        "d1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_femto_d1_in1k.weights.h5",
        }
    },
    "ConvNeXtPico": {
        "d1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_pico_d1_in1k.weights.h5",
        }
    },
    "ConvNeXtNano": {
        "d1h_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_nano_d1h_in1k.weights.h5",
        },
        "in12k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_nano_in12k_ft_in1k.weights.h5",
        },
    },
    # ConvNeXtV1
    "ConvNeXtTiny": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_tiny_fb_in1k.weights.h5",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_tiny_fb_in22k.weights.h5",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_tiny_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_tiny_fb_in22k_ft_in1k_384.weights.h5",
        },
    },
    "ConvNeXtSmall": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_small_fb_in1k.weights.h5",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_small_fb_in22k.weights.h5",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_small_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_small_fb_in22k_ft_in1k_384.weights.h5",
        },
    },
    "ConvNeXtBase": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_base_fb_in1k.weights.h5",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_base_fb_in22k.weights.h5",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_base_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_base_fb_in22k_ft_in1k_384.weights.h5",
        },
    },
    "ConvNeXtLarge": {
        "fb_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_large_fb_in1k.weights.h5",
        },
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_large_fb_in22k.weights.h5",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_large_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_large_fb_in22k_ft_in1k_384.weights.h5",
        },
    },
    "ConvNeXtXLarge": {
        "fb_in22k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_xlarge_fb_in22k.weights.h5",
        },
        "fb_in22k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_xlarge_fb_in22k_ft_in1k.weights.h5",
        },
        "fb_in22k_ft_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnext_xlarge_fb_in22k_ft_in1k_384.weights.h5",
        },
    },
}
