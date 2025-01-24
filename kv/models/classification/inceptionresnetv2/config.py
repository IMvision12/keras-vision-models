INCEPTIONRESNETV2_MODEL_CONFIG = {}  # No config Required for InceptionResNetV2

INCEPTIONRESNETV2_WEIGHTS_CONFIG = {
    "InceptionResNetV2": {
        "tf_ens_adv_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/inception_v3_gluon_in1k.keras",
        },
        "tf_in1k": {
            "url": "https://github.com/IMvision12/keras-vision/releases/download/v0.1/inception_v3_tf_in1k.keras",
        },
    },
}
