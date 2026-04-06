"""Sam3Video configuration."""

SAM3_VIDEO_MODEL_CONFIG = {
    "Sam3Video": {
        # Vision neck (FPN bridging detector to tracker)
        "backbone_hidden_size": 1024,
        "fpn_hidden_size": 256,
        "fpn_scale_factors": [4.0, 2.0, 1.0, 0.5],
        # Detection thresholds
        "score_threshold_detection": 0.5,
        "det_nms_thresh": 0.1,
        "new_det_thresh": 0.7,
        # Association thresholds
        "assoc_iou_thresh": 0.1,
        "trk_assoc_iou_thresh": 0.5,
        # Reconditioning
        "recondition_on_trk_masks": True,
        "recondition_every_nth_frame": 16,
        "high_conf_thresh": 0.8,
        "high_iou_thresh": 0.8,
        # Hotstart
        "hotstart_delay": 15,
        "hotstart_unmatch_thresh": 8,
        "hotstart_dup_thresh": 8,
        # Track management
        "init_trk_keep_alive": 30,
        "max_trk_keep_alive": 30,
        "min_trk_keep_alive": -1,
        # Other
        "low_res_mask_size": 288,
        "fill_hole_area": 16,
        "max_num_objects": 10000,
        "suppress_overlapping_based_on_recent_occlusion_threshold": 0.7,
    }
}

SAM3_VIDEO_WEIGHTS_CONFIG = {
    "Sam3Video": {
        "pcs": {
            "url": "",
        }
    }
}
