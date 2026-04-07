"""Unified weight loading for all SAM3 model types.

Sam3VideoModel is the canonical format (like HF's facebook/sam3 checkpoint).
One file, one save_weights/load_weights call, no duplication.

Usage:
    from kmodels.models.sam3.sam3_model import Sam3
    from kmodels.models.sam3_tracker_video.sam3_tracker_video_model import Sam3TrackerVideo
    from kmodels.models.sam3_video.sam3_video_model import Sam3Video

    sam3 = Sam3(input_shape=(1008, 1008, 3), weights=None)
    tv = Sam3TrackerVideo(sam3_model=sam3, weights=None)
    vm = Sam3Video(sam3_model=sam3, tracker_video_model=tv, weights=None)
    vm.load_weights("sam3_unified.weights.h5")
    # All components now have correct weights.
"""
