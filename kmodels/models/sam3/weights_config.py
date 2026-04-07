"""Shared weight configuration for all SAM3 model types.

All 4 models (Sam3, Sam3Tracker, Sam3TrackerVideo, Sam3Video) load from
the same unified weight file — matching HF's facebook/sam3 checkpoint.

Sam3VideoModel is the canonical format. Its save_weights/load_weights
contains all components: detector + tracker + video neck.
"""

# Single URL for the unified weight file.
SAM3_UNIFIED_WEIGHTS_URL = "https://github.com/IMvision12/keras-models/releases/download/sam3-v1.0/sam3_unified.weights.h5"

SAM3_UNIFIED_WEIGHTS_CONFIG = {
    "pcs": {
        "url": SAM3_UNIFIED_WEIGHTS_URL,
    },
}


def load_unified_weights(
    sam3_model, tracker_video_model=None, video_model=None, weights="pcs"
):
    """Load weights from the unified file into all SAM3 components.

    Downloads the unified weight file if needed, then loads into the
    Sam3VideoModel (which owns all sub-models).

    Args:
        sam3_model: SAM3Model instance (detector + text_encoder + geo_encoder).
        tracker_video_model: Sam3TrackerVideoModel instance (optional).
        video_model: Sam3VideoModel instance (optional).
        weights: weight variant name ("pcs") or file path.

    If only sam3_model is provided, builds the full Sam3VideoModel internally
    just for weight loading, then discards the tracker/video components.
    """
    from kmodels.utils import download_file

    # Resolve weight file path
    if weights in SAM3_UNIFIED_WEIGHTS_CONFIG:
        url = SAM3_UNIFIED_WEIGHTS_CONFIG[weights]["url"]
        filepath = download_file(url)
    else:
        filepath = weights  # Assume it's a file path

    # Build the full Sam3VideoModel if needed (canonical format for weights)
    if video_model is not None:
        video_model.load_weights(filepath)
    elif tracker_video_model is not None:
        # Build a temporary video model to load the full file
        from kmodels.models.sam3_video.config import SAM3_VIDEO_MODEL_CONFIG
        from kmodels.models.sam3_video.sam3_video_model import Sam3VideoModel

        vm = Sam3VideoModel(
            sam3_model=sam3_model,
            tracker_video_model=tracker_video_model,
            video_config=SAM3_VIDEO_MODEL_CONFIG["Sam3Video"],
        )
        vm.load_weights(filepath, skip_mismatch=True)
    else:
        # Only detector needed — build full pipeline just for loading
        from kmodels.models.sam3_tracker_video.sam3_tracker_video_model import (
            Sam3TrackerVideoModel,
        )
        from kmodels.models.sam3_video.config import SAM3_VIDEO_MODEL_CONFIG
        from kmodels.models.sam3_video.sam3_video_model import Sam3VideoModel

        tv = Sam3TrackerVideoModel(sam3_model=sam3_model)
        vm = Sam3VideoModel(
            sam3_model=sam3_model,
            tracker_video_model=tv,
            video_config=SAM3_VIDEO_MODEL_CONFIG["Sam3Video"],
        )
        vm.load_weights(filepath, skip_mismatch=True)
