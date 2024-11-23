RESNET_MODEL_CONFIG = {
    "resnet50": {
        "block_type": "bottleneck",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet101": {
        "block_type": "bottleneck",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet152": {
        "block_type": "bottleneck",
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnext50_32x4d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
    },
    "resnext101_32x4d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
    },
    "resnext101_32x8d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
    },
    "resnext101_32x16d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 8,
    },
    "resnext101_32x32d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 16,
    },
    "seresnet50": {
        "block_type": "bottleneck",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "senet": True,
    },
    "seresnext50_32x4d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
    },
    "seresnext101_32x4d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
    },
    "seresnext101_32x8d": {
        "block_type": "resnext",
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "senet": True,
    },
}

RESNET_WEIGHTS_CONFIG = {
    # ResNet Variants
    "ResNet50": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "ResNet101": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet101_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet101_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "ResNet152": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet152_tv_in1k.keras",
        },
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet152_a1_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    # ResNeXt Variants
    "ResNeXt50_32x4d": {
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext50_32x4d_a1_in1k.keras",
        },
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext50_32x4d_tv_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
        "fb_ssl_yfcc100m_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext50_32x4d_fb_ssl_yfcc100m_ft_in1k.keras",
        },
        "fb_swsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext50_32x4d_fb_swsl_ig1b_ft_in1k.keras",
        },
    },
    "ResNeXt101_32x4d": {
        "fb_ssl_yfcc100m_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x4d_fb_ssl_yfcc100m_ft_in1k.keras",
        },
        "fb_swsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x4d_fb_swsl_ig1b_ft_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "ResNeXt101_32x8d": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x8d_tv_in1k.keras",
        },
        "fb_wsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x8d_fb_wsl_ig1b_ft_in1k.keras",
        },
        "fb_ssl_yfcc100m_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x8d_fb_ssl_yfcc100m_ft_in1k.keras",
        },
        "fb_swsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x8d_fb_swsl_ig1b_ft_in1k.keras",
        },
    },
    "ResNeXt101_32x16d": {
        "fb_wsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x16d_fb_wsl_ig1b_ft_in1k.keras",
        },
        "fb_ssl_yfcc100m_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x16d_fb_ssl_yfcc100m_ft_in1k.keras",
        },
        "fb_swsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x16d_fb_swsl_ig1b_ft_in1k.keras",
        },
    },
    "ResNeXt101_32x32d": {
        "fb_wsl_ig1b_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnext101_32x32d_fb_wsl_ig1b_ft_in1k.keras",
        }
    },
    # SE-ResNet and SE-ResNeXt Variants
    "SEResNet50": {
        "a1_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnet50_a1_in1k.keras",
        }
    },
    "SEResNeXt50_32x4d": {
        "racm_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnext50_32x4d_racm_in1k.keras",
        },
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "SEResNeXt101_32x4d": {
        "gluon_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/resnet50_a1_in1k.keras",
        },
    },
    "SEResNeXt101_32x8d": {
        "ah_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/seresnext101_32x8d_ah_in1k.keras",
        }
    },
}
