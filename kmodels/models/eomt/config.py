EOMT_MODEL_CONFIG = {
    "EoMTSmall": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_blocks": 3,
        "layerscale_value": 1.0,
    },
    "EoMTBase": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_blocks": 3,
        "layerscale_value": 1.0,
    },
    "EoMTLarge": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_blocks": 4,
        "layerscale_value": 1e-5,
    },
}

EOMT_WEIGHTS_CONFIG = {
    "EoMTSmall": {
        "coco_panoptic_640": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EoMT/coco_panoptic_eomt_small_640_2x.weights.h5",
        },
    },
    "EoMTBase": {
        "coco_panoptic_640": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EoMT/coco_panoptic_eomt_base_640_2x.weights.h5",
        },
    },
    "EoMTLarge": {
        "coco_panoptic_640": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EoMT/coco_panoptic_eomt_large_640.weights.h5",
        },
        "coco_instance_640": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EoMT/coco_instance_eomt_large_640.weights.h5",
        },
        "ade20k_semantic_512": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/EoMT/ade20k_semantic_eomt_large_512.weights.h5",
        },
    },
}
