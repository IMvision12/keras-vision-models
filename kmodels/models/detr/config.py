DETR_MODEL_CONFIG = {
    "DETRResNet50": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout_rate": 0.1,
        "backbone_variant": "ResNet50",
    },
    "DETRResNet101": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout_rate": 0.1,
        "backbone_variant": "ResNet101",
    },
}

DETR_WEIGHTS_CONFIG = {
    "DETRResNet50": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/DeTR/detr_resnet_50_coco.weights.h5",
        },
    },
    "DETRResNet101": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/DeTR/detr_resnet_101_coco.weights.h5",
        },
    },
}
