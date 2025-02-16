RESNETV2_MODEL_CONFIG = {
    "ResNetV2_50": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 1,
        "use_batchnorm": True,
        "use_conv": True,
    },
    "ResNetV2_50x1": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 1,
        "use_batchnorm": False,
        "use_conv": False,
    },
    "ResNetV2_50x3": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 3,
        "use_batchnorm": False,
        "use_conv": False,
    },
    "ResNetV2_101": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 1,
        "use_batchnorm": True,
        "use_conv": True,
    },
    "ResNetV2_101x1": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 1,
        "use_batchnorm": False,
        "use_conv": False,
    },
    "ResNetV2_101x3": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 3,
        "use_batchnorm": False,
        "use_conv": False,
    },
    "ResNetV2_152x2": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 2,
        "use_batchnorm": False,
        "use_conv": False,
    },
    "ResNetV2_152x4": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 4,
        "use_batchnorm": False,
        "use_conv": False,
    },
}

RESNETV2_WEIGHTS_CONFIG = {
    "ResNetV2_50": {
        "a1h_in1k": {
            "url": "",
        },
    },
    "ResNetV2_50x1": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
    "ResNetV2_50x3": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
    "ResNetV2_101": {
        "a1h_in1k": {
            "url": "",
        },
    },
    "ResNetV2_101x1": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
    "ResNetV2_101x3": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
    "ResNetV2_152x2": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
    "ResNetV2_152x4": {
        "goog_in21k_ft_in1k": {
            "url": "",
        },
        "goog_in21k": {
            "url": "",
        },
    },
}
