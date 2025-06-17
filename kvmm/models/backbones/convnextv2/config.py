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
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_atto_fcmae_ft_in1k.weights.h5",
        }
    },
    "ConvNeXtV2Femto": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_femto_fcmae_ft_in1k.weights.h5",
        }
    },
    "ConvNeXtV2Pico": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_pico_fcmae_ft_in1k.weights.h5",
        }
    },
    "ConvNeXtV2Nano": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_nano_fcmae_ft_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in22k_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_nano_fcmae_ft_in22k_in1k_384.weights.h5",
        },
    },
    "ConvNeXtV2Tiny": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in22k_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_tiny_fcmae_ft_in22k_in1k_384.weights.h5",
        },
    },
    "ConvNeXtV2Base": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_base_fcmae_ft_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_base_fcmae_ft_in22k_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_base_fcmae_ft_in22k_in1k_384.weights.h5",
        },
    },
    "ConvNeXtV2Large": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_large_fcmae_ft_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_large_fcmae_ft_in22k_in1k.weights.h5",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_large_fcmae_ft_in22k_in1k_384.weights.h5",
        },
    },
    "ConvNeXtV2Huge": {
        "fcmae_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_huge_fcmae_ft_in1k.weights.json",
        },
        "fcmae_ft_in22k_in1k_384": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_huge_fcmae_ft_in22k_in1k_384.weights.json",
        },
        "fcmae_ft_in22k_in1k_512": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convnextv2_huge_fcmae_ft_in22k_in1k_512.weights.json",
        },
    },
}
