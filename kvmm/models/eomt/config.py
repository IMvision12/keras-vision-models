EOMT_MODEL_CONFIG = {
    "EoMT_Small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "mlp_ratio": 4,
        "patch_size": 16,
        "num_register_tokens": 4,
        "num_blocks": 3,
        "num_upscale_blocks": 2,
        "layerscale_value": 1.0,
        "use_swiglu_ffn": False,
    },
    "EoMT_Base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "mlp_ratio": 4,
        "patch_size": 16,
        "num_register_tokens": 4,
        "num_blocks": 3,
        "num_upscale_blocks": 2,
        "layerscale_value": 1.0,
        "use_swiglu_ffn": False,
    },
    "EoMT_Large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 16,
        "num_register_tokens": 4,
        "num_blocks": 4,
        "num_upscale_blocks": 2,
        "layerscale_value": 1e-5,
        "use_swiglu_ffn": False,
    },
}

EOMT_WEIGHTS_CONFIG = {
    "EoMT_Small": {
        "coco_panoptic_640": {
            "url": "",
        },
    },
    "EoMT_Base": {
        "coco_panoptic_640": {
            "url": "",
        },
    },
    "EoMT_Large": {
        "coco_panoptic_640": {
            "url": "",
        },
        "coco_instance_640": {
            "url": "",
        },
        "ade20k_semantic_512": {
            "url": "",
        },
    },
}
