SAM2_MODEL_CONFIG = {
    "Sam2Tiny": {
        "hidden_size": 96,
        "blocks_per_stage": [1, 2, 7, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [5, 7, 9],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "Sam2Small": {
        "hidden_size": 96,
        "blocks_per_stage": [1, 2, 11, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [7, 10, 13],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "Sam2BasePlus": {
        "hidden_size": 112,
        "blocks_per_stage": [2, 3, 16, 3],
        "embed_dim_per_stage": [112, 224, 448, 896],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [12, 16, 20],
        "backbone_channel_list": [896, 448, 224, 112],
        "window_pos_embed_bg_size": [14, 14],
    },
    "Sam2Large": {
        "hidden_size": 144,
        "blocks_per_stage": [2, 6, 36, 4],
        "embed_dim_per_stage": [144, 288, 576, 1152],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 16, 8],
        "global_attention_blocks": [23, 33, 43],
        "backbone_channel_list": [1152, 576, 288, 144],
    },
}

SAM2_WEIGHTS_CONFIG = {
    "Sam2Tiny": {
        "sav": {
            "url": "",
        },
    },
    "Sam2Small": {
        "sav": {
            "url": "",
        },
    },
    "Sam2BasePlus": {
        "sav": {
            "url": "",
        },
    },
    "Sam2Large": {
        "sav": {
            "url": "",
        },
    },
}
