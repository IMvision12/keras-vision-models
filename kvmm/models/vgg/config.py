VGG_MODEL_CONFIG = {
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

VGG_WEIGHTS_CONFIG = {
    "VGG16": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/vgg16_tv_in1k.weights.h5"
        },
    },
    "VGG19": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/vgg19_tv_in1k.weights.h5"
        },
    },
    "VGG16_BN": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/vgg16_bn_tv_in1k.weights.h5"
        },
    },
    "VGG19_BN": {
        "tv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/vgg19_bn_tv_in1k.weights.h5"
        },
    },
}
