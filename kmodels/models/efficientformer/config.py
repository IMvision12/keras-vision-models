EFFICIENTFORMER_MODEL_CONFIG = {
    "l1": {
        "depths": [3, 2, 6, 4],
        "embed_dims": [48, 96, 224, 448],
        "num_vit": 1,
    },
    "l3": {
        "depths": [4, 4, 12, 6],
        "embed_dims": [64, 128, 320, 512],
        "num_vit": 4,
    },
    "l7": {
        "depths": [6, 6, 18, 8],
        "embed_dims": [96, 192, 384, 768],
        "num_vit": 8,
    },
}

EFFICIENTFORMER_WEIGHTS_CONFIG = {
    "EfficientFormerL1": {
        "snap_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l1_snap_dist_in1k.weights.h5",
        },
    },
    "EfficientFormerL3": {
        "snap_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l3_snap_dist_in1k.weights.h5",
        },
    },
    "EfficientFormerL7": {
        "snap_dist_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l7_snap_dist_in1k.weights.h5",
        },
    },
}
