"""Sam3TrackerVideo configuration."""

SAM3_TRACKER_VIDEO_MEMORY_ATTENTION_CONFIG = {
    "hidden_size": 256,
    "num_layers": 4,
    "num_attention_heads": 1,
    "downsample_rate": 1,
    "feed_forward_hidden_size": 2048,
    "feed_forward_hidden_act": "relu",
    "dropout": 0.1,
    "rope_theta": 10000,
    "rope_feat_sizes": [72, 72],
    "rope_dropout": 0.1,
}

SAM3_TRACKER_VIDEO_MEMORY_ENCODER_CONFIG = {
    "hidden_size": 256,
    "output_channels": 64,
    # Mask downsampler
    "mask_downsampler_embed_dim": 256,
    "mask_downsampler_kernel_size": 3,
    "mask_downsampler_stride": 2,
    "mask_downsampler_padding": 1,
    "mask_downsampler_total_stride": 16,
    "mask_downsampler_hidden_act": "gelu",
    # Memory fuser (ConvNeXt blocks)
    "memory_fuser_num_layers": 2,
    "memory_fuser_embed_dim": 256,
    "memory_fuser_intermediate_dim": 1024,
    "memory_fuser_kernel_size": 7,
    "memory_fuser_padding": 3,
    "memory_fuser_layer_scale_init_value": 1e-6,
    "memory_fuser_hidden_act": "gelu",
}

SAM3_TRACKER_VIDEO_MODEL_CONFIG = {
    "Sam3TrackerVideo": {
        "prompt_encoder": {
            "hidden_size": 256,
            "image_size": 1008,
            "patch_size": 14,
            "mask_input_channels": 16,
            "num_point_embeddings": 4,
            "scale": 1,
        },
        "mask_decoder": {
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
        },
        "memory_attention": SAM3_TRACKER_VIDEO_MEMORY_ATTENTION_CONFIG,
        "memory_encoder": SAM3_TRACKER_VIDEO_MEMORY_ENCODER_CONFIG,
        # Model-level params
        "hidden_dim": 256,
        "mem_dim": 64,
        "num_maskmem": 7,
        "max_object_pointers_in_encoder": 16,
        "max_cond_frame_num": 4,
        "enable_temporal_pos_encoding_for_object_pointers": True,
        "enable_occlusion_spatial_embedding": True,
    }
}

SAM3_TRACKER_VIDEO_WEIGHTS_CONFIG = {
    "Sam3TrackerVideo": {
        "pcs": {
            "url": "",
        }
    }
}
