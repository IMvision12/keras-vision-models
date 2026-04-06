"""Sam3Tracker configuration."""

SAM3_TRACKER_PROMPT_ENCODER_CONFIG = {
    "hidden_size": 256,
    "image_size": 1008,
    "patch_size": 14,
    "mask_input_channels": 16,
    "num_point_embeddings": 4,
    "scale": 1,
}

SAM3_TRACKER_MASK_DECODER_CONFIG = {
    "hidden_size": 256,
    "mlp_dim": 2048,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "attention_downsample_rate": 2,
    "num_multimask_outputs": 3,
    "iou_head_depth": 3,
    "iou_head_hidden_dim": 256,
    "dynamic_multimask_via_stability": True,
    "dynamic_multimask_stability_delta": 0.05,
    "dynamic_multimask_stability_thresh": 0.98,
}

SAM3_TRACKER_MODEL_CONFIG = {
    "Sam3Tracker": {
        "prompt_encoder": SAM3_TRACKER_PROMPT_ENCODER_CONFIG,
        "mask_decoder": SAM3_TRACKER_MASK_DECODER_CONFIG,
    }
}

SAM3_TRACKER_WEIGHTS_CONFIG = {
    "Sam3Tracker": {
        "pcs": {
            "url": "",
        }
    }
}
