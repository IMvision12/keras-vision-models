"""Sam3Video configuration."""

from kmodels.models.sam3.weights_config import SAM3_UNIFIED_WEIGHTS_CONFIG

SAM3_VIDEO_MODEL_CONFIG = {
    "Sam3Video": {
        "backbone_hidden_size": 1024,
        "fpn_hidden_size": 256,
        "fpn_scale_factors": [4.0, 2.0, 1.0, 0.5],
        "score_threshold_detection": 0.5,
        "det_nms_thresh": 0.1,
        "new_det_thresh": 0.7,
        "assoc_iou_thresh": 0.1,
        "trk_assoc_iou_thresh": 0.5,
        "recondition_on_trk_masks": True,
        "recondition_every_nth_frame": 16,
        "high_conf_thresh": 0.8,
        "high_iou_thresh": 0.8,
        "hotstart_delay": 15,
        "hotstart_unmatch_thresh": 8,
        "hotstart_dup_thresh": 8,
        "init_trk_keep_alive": 30,
        "max_trk_keep_alive": 30,
        "min_trk_keep_alive": -1,
        "low_res_mask_size": 288,
        "fill_hole_area": 16,
        "max_num_objects": 10000,
        "suppress_overlapping_based_on_recent_occlusion_threshold": 0.7,
    }
}

SAM3_VIDEO_WEIGHTS_CONFIG = {
    "Sam3Video": SAM3_UNIFIED_WEIGHTS_CONFIG,
}
