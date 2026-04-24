QWEN2_VL_MODEL_CONFIG = {
    "Qwen2VL2B": {
        "text_config": {
            "vocab_size": 151936,
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "mrope_section": [16, 24, 24],
            "tie_word_embeddings": True,
        },
        "vision_config": {
            "depth": 32,
            "embed_dim": 1280,
            "hidden_size": 1536,
            "mlp_ratio": 4,
            "num_heads": 16,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    },
}


# Qwen2-VL weights are NOT redistributed by kmodels: we download from the
# official Qwen HF repos on first use, convert on the fly, and cache under
# ``~/.cache/kmodels/<variant>/``. Subsequent loads are instant.
QWEN2_VL_HF_CONVERT_VARIANTS = {
    "Qwen2VL2B": "Qwen/Qwen2-VL-2B-Instruct",
}

QWEN2_VL_HF_CONVERT_DEFAULT_ALIAS = {
    "Qwen2VL2B": "qwen",
}
