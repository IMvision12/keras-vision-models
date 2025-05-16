EFFICIENTNETV2_BLOCK_CONFIG = {
    "EfficientNetV2S": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 48,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 6,
            "input_filters": 64,
            "output_filters": 128,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 9,
            "input_filters": 128,
            "output_filters": 160,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 15,
            "input_filters": 160,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2M": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2L": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2XL": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 16,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 24,
            "input_filters": 192,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 32,
            "input_filters": 256,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 512,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    # For all B variants B0, B1, B2, B3
    "EfficientNetV2B": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

EFFICIENTNETV2_MODEL_CONFIG = {
    "EfficientNetV2S": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 300,
    },
    "EfficientNetV2M": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
    },
    "EfficientNetV2L": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
    },
    "EfficientNetV2B0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 192,
    },
    "EfficientNetV2B1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "default_size": 192,
    },
    "EfficientNetV2B2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "default_size": 208,
    },
    "EfficientNetV2B3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "default_size": 240,
    },
    "EfficientNetV2XL": {  # only for 21k pretraining
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
    },
}

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

EFFICIENTNETV2_WEIGHTS_CONFIG = {
    "EfficientNetV2S": {
        "in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_s_in21k_ft_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_s_in1k.weights.h5",
        },
        "in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_s_in21k.weights.h5",
        },
    },
    "EfficientNetV2M": {
        "in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_m_in21k_ft_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_m_in1k.weights.h5",
        },
        "in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_m_in21k.weights.h5",
        },
    },
    "EfficientNetV2L": {
        "in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_l_in21k_ft_in1k.weights.h5",
        },
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_l_in1k.weights.h5",
        },
        "in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_l_in21k.weights.h5",
        },
    },
    "EfficientNetV2XL": {
        "in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_xl_in21k_ft_in1k.weights.h5",
        },
        "in21k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_xl_in21k.weights.h5",
        },
    },
    "EfficientNetV2B0": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_b0_in1k.weights.h5",
        },
    },
    "EfficientNetV2B1": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_b1_in1k.weights.h5",
        },
    },
    "EfficientNetV2B2": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_b2_in1k.weights.h5",
        },
    },
    "EfficientNetV2B3": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_b3_in1k.weights.h5",
        },
        "in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.2/tf_efficientnetv2_b3_in21k_ft_in1k.weights.h5",
        },
    },
}
