RT_DETR_MODEL_CONFIG = {
    "RTDETRResNet18": {
        "backbone_hidden_sizes": [64, 128, 256, 512],
        "backbone_block_repeats": [2, 2, 2, 2],
        "backbone_layer_type": "basic",
        "encoder_in_channels": [128, 256, 512],
        "hidden_expansion": 0.5,
        "decoder_layers": 3,
    },
    "RTDETRResNet34": {
        "backbone_hidden_sizes": [64, 128, 256, 512],
        "backbone_block_repeats": [3, 4, 6, 3],
        "backbone_layer_type": "basic",
        "encoder_in_channels": [128, 256, 512],
        "hidden_expansion": 0.5,
        "decoder_layers": 4,
    },
    "RTDETRResNet50": {
        "backbone_hidden_sizes": [256, 512, 1024, 2048],
        "backbone_block_repeats": [3, 4, 6, 3],
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": [512, 1024, 2048],
        "decoder_layers": 6,
    },
    "RTDETRResNet101": {
        "backbone_hidden_sizes": [256, 512, 1024, 2048],
        "backbone_block_repeats": [3, 4, 23, 3],
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": [512, 1024, 2048],
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_layers": 6,
    },
}

RT_DETR_WEIGHTS_CONFIG = {
    "RTDETRResNet18": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r18vd_coco.weights.h5"
        },
        "coco_o365": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r18vd_coco_o365.weights.h5"
        },
    },
    "RTDETRResNet34": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r34vd_coco.weights.h5"
        },
    },
    "RTDETRResNet50": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r50vd_coco.weights.h5"
        },
        "coco_o365": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r50vd_coco_o365.weights.h5"
        },
    },
    "RTDETRResNet101": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r101vd_coco.weights.h5"
        },
        "coco_o365": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/RT-DETR/rtdetr_r101vd_coco_o365.weights.h5"
        },
    },
}
