DEPTH_ANYTHING_V1_MODEL_CONFIG = {
    "DepthAnythingV1Small": {
        "backbone_dim": 384,
        "backbone_depth": 12,
        "backbone_num_heads": 6,
        "out_indices": [9, 10, 11, 12],
        "neck_hidden_sizes": [48, 96, 192, 384],
        "fusion_hidden_size": 64,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
    "DepthAnythingV1Base": {
        "backbone_dim": 768,
        "backbone_depth": 12,
        "backbone_num_heads": 12,
        "out_indices": [9, 10, 11, 12],
        "neck_hidden_sizes": [96, 192, 384, 768],
        "fusion_hidden_size": 128,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
    "DepthAnythingV1Large": {
        "backbone_dim": 1024,
        "backbone_depth": 24,
        "backbone_num_heads": 16,
        "out_indices": [21, 22, 23, 24],
        "neck_hidden_sizes": [256, 512, 1024, 1024],
        "fusion_hidden_size": 256,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
}

DEPTH_ANYTHING_V1_WEIGHTS_CONFIG = {
    "DepthAnythingV1Small": {
        "da_v1": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything/depth_anything_v1_small.weights.h5",
        },
    },
    "DepthAnythingV1Base": {
        "da_v1": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything/depth_anything_v1_base.weights.h5",
        },
    },
    "DepthAnythingV1Large": {
        "da_v1": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/depth_anything/depth_anything_v1_large.weights.h5",
        },
    },
}
