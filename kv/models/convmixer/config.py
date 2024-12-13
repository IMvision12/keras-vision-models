CONVMIXER_MODEL_CONFIG = {
    "ConvMixer_1536_20": {
        "dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
    },
    "ConvMixer_768_32": {
        "dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
    },
    "ConvMixer_1024_20": {
        "dim": 1024,
        "depth": 20,
        "patch_size": 14,
        "kernel_size": 9,
    },
}

CONVMIXER_WEIGHTS_CONFIG = {
    "ConvMixer_1536_20": {
        "in1k": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convmixer_1536_20_in1k.keras",
    },
    "ConvMixer_768_32": {
        "in1k": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convmixer_768_32_in1k.keras",
    },
    "ConvMixer_1024_20": {
        "in1k": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/convmixer_1024_20_ks9_p14_in1k.keras",
    },
}
