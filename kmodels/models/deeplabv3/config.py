DEEPLABV3_MODEL_CONFIG = {
    "DeepLabV3ResNet50": {
        "backbone_variant": "ResNet50",
        "num_classes": 21,
    },
    "DeepLabV3ResNet101": {
        "backbone_variant": "ResNet101",
        "num_classes": 21,
    },
}

DEEPLABV3_WEIGHTS_CONFIG = {
    "DeepLabV3ResNet50": {
        "coco_voc": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/deeplabv3/deeplabv3_resnet50_coco_voc.weights.h5",
        },
    },
    "DeepLabV3ResNet101": {
        "coco_voc": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/deeplabv3/deeplabv3_resnet101_coco_voc.weights.h5",
        },
    },
}
