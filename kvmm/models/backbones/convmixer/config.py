CONVMIXER_MODEL_CONFIG = {
    "ConvMixer1536D20": {
        "dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
    },
    "ConvMixer768D32": {
        "dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
    },
    "ConvMixer1024D20": {
        "dim": 1024,
        "depth": 20,
        "patch_size": 14,
        "kernel_size": 9,
    },
}

CONVMIXER_WEIGHTS_CONFIG = {
    "ConvMixer1536D20": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convmixer_1536_20_in1k.weights.h5"
        },
    },
    "ConvMixer768D32": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convmixer_768_32_in1k.weights.h5"
        },
    },
    "ConvMixer1024D20": {
        "in1k": {
            "url": "https://github.com/IMvision12/keras-vision-models/releases/download/v0.1/convmixer_1024_20_ks9_p14_in1k.weights.h5"
        },
    },
}
