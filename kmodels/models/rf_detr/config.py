RF_DETR_MODEL_CONFIG = {
    "RFDETRNano": {
        "positional_encoding_size": 24,
        "resolution": 384,
        "dec_layers": 2,
    },
    "RFDETRSmall": {
        "positional_encoding_size": 32,
        "resolution": 512,
        "dec_layers": 3,
    },
    "RFDETRMedium": {
        "positional_encoding_size": 36,
        "resolution": 576,
        "dec_layers": 4,
    },
    "RFDETRBase": {
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 14,
        "num_windows": 4,
        "positional_encoding_size": 37,
        "resolution": 560,
        "dec_layers": 3,
    },
    "RFDETRLarge": {
        "positional_encoding_size": 44,
        "resolution": 704,
        "dec_layers": 4,
    },
}

RF_DETR_WEIGHTS_CONFIG = {
    "RFDETRNano": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/rf-detr/rf_detr_nano_coco.weights.h5",
        },
    },
    "RFDETRSmall": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/rf-detr/rf_detr_small_coco.weights.h5",
        },
    },
    "RFDETRMedium": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/rf-detr/rf_detr_medium_coco.weights.h5",
        },
    },
    "RFDETRBase": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/rf-detr/rf_detr_base_coco.weights.h5",
        },
    },
    "RFDETRLarge": {
        "coco": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/rf-detr/rf_detr_large_coco.weights.h5",
        },
    },
}
