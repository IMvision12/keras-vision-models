EFFICIENTNET_MODEL_CONFIG = {
    "EfficientNetB0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 224,
        "dropout_rate": 0.2,
    },
    "EfficientNetB1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "default_size": 240,
        "dropout_rate": 0.2,
    },
    "EfficientNetB2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "default_size": 260,
        "dropout_rate": 0.3,
    },
    "EfficientNetB3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "default_size": 300,
        "dropout_rate": 0.3,
    },
    "EfficientNetB4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "default_size": 380,
        "dropout_rate": 0.4,
    },
    "EfficientNetB5": {
        "width_coefficient": 1.6,
        "depth_coefficient": 2.2,
        "default_size": 456,
        "dropout_rate": 0.4,
    },
    "EfficientNetB6": {
        "width_coefficient": 1.8,
        "depth_coefficient": 2.6,
        "default_size": 528,
        "dropout_rate": 0.5,
    },
    "EfficientNetB7": {
        "width_coefficient": 2.0,
        "depth_coefficient": 3.1,
        "default_size": 600,
        "dropout_rate": 0.5,
    },
    "EfficientNetB8": {
        "width_coefficient": 2.2,
        "depth_coefficient": 3.6,
        "default_size": 672,
        "dropout_rate": 0.5,
    },
    "EfficientNetL2": {
        "width_coefficient": 4.3,
        "depth_coefficient": 5.3,
        "default_size": 800,
        "dropout_rate": 0.5,
    },
}


DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

EFFICIENTNET_WEIGHTS_CONFIG = {
    "EfficientNetB0": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b0_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b0_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b0_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b0_in1k.weights.h5",
        },
    },
    "EfficientNetB1": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b1_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b1_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b1_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b1_in1k.weights.h5",
        },
    },
    "EfficientNetB2": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b2_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b2_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b2_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b2_in1k.weights.h5",
        },
    },
    "EfficientNetB3": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b3_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b3_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b3_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b3_in1k.weights.h5",
        },
    },
    "EfficientNetB4": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b4_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b4_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b4_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b4_in1k.weights.h5",
        },
    },
    "EfficientNetB5": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b5_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b5_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b5_aa_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b5_in1k.weights.h5",
        },
    },
    "EfficientNetB6": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b6_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b6_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b6_aa_in1k.weights.h5",
        },
    },
    "EfficientNetB7": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b7_ns_jft_in1k.weights.h5",
        },
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b7_ap_in1k.weights.h5",
        },
        "aa_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b7_aa_in1k.weights.h5",
        },
    },
    "EfficientNetB8": {
        "ap_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_b8_ap_in1k.weights.h5",
        },
    },
    "EfficientNetL2": {
        "ns_jft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_l2_ns_jft_in1k.weights.h5",
        },
        "ns_jft_in1k_475": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_l2_ns_jft_in1k_475.weights.h5",
        },
    },
}
