DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}

EFFICIENTNET_LITE_MODEL_CONFIG = {
    "EfficientNetLite0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 224,
        "dropout_rate": 0.2,
    },
    "EfficientNetLite1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "default_size": 240,
        "dropout_rate": 0.2,
    },
    "EfficientNetLite2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "default_size": 260,
        "dropout_rate": 0.3,
    },
    "EfficientNetLite3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "default_size": 300,
        "dropout_rate": 0.3,
    },
    "EfficientNetLite4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "default_size": 380,
        "dropout_rate": 0.3,
    },
}

EFFICIENTNET_LITE_WEIGHTS_CONFIG = {
    "EfficientNetLite0": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_lite0_in1k.weights.h5",
        }
    },
    "EfficientNetLite1": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_lite1_in1k.weights.h5",
        },
    },
    "EfficientNetLite2": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_lite2_in1k.weights.h5",
        },
    },
    "EfficientNetLite3": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_lite3_in1k.weights.h5",
        },
    },
    "EfficientNetLite4": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/tf_efficientnet_lite4_in1k.weights.h5",
        },
    },
}
