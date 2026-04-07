"""Sam3Video model: detector + tracker + vision neck orchestration.

Combines the SAM3 detector (text-conditioned detection) with the
Sam3TrackerVideo (memory-based tracking) via a Sam3VisionNeck FPN
that bridges detector backbone features to tracker format.

The main inference loop:
1. Extract vision features once per frame
2. Run detector on all text prompts (reuses vision features)
3. Bridge features to tracker via vision neck
4. Propagate tracker states (memory-conditioned)
5. Associate detections with tracked objects
6. Update memory with new masks
"""

from collections import OrderedDict, defaultdict

import keras
import numpy as np
from keras import ops

from kmodels.model_registry import register_model

from .config import SAM3_VIDEO_MODEL_CONFIG, SAM3_VIDEO_WEIGHTS_CONFIG
from .sam3_video_layers import Sam3VisionNeck


class Sam3VideoInferenceSession:
    """Manages video inference state: frame storage, object tracking, memory.

    Holds all per-object tracking state, memory buffers, and caches
    needed for multi-frame video inference.
    """

    def __init__(
        self,
        video_height=None,
        video_width=None,
        max_vision_features_cache_size=1,
    ):
        self.video_height = video_height
        self.video_width = video_width
        self.max_vision_cache = max_vision_features_cache_size

        # Frame storage
        self.processed_frames = {}
        self.num_frames = 0

        # Object ID management
        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []
        self.max_obj_id = 0

        # Per-object prompts and inputs
        self.point_inputs_per_obj = defaultdict(dict)
        self.mask_inputs_per_obj = defaultdict(dict)

        # Per-object model outputs
        # {obj_idx: {"cond_frame_outputs": {frame_idx: out}, "non_cond_frame_outputs": {frame_idx: out}}}
        self.output_dict_per_obj = defaultdict(
            lambda: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        )
        self.frames_tracked_per_obj = defaultdict(dict)

        # Multi-prompt support
        self.prompts = {}  # prompt_id → text
        self.prompt_embeddings = {}  # prompt_id → embedding
        self.prompt_attention_masks = {}
        self.obj_id_to_prompt_id = {}
        self._next_prompt_id = 0

        # Tracking metadata
        self.obj_id_to_score = {}
        self.obj_id_to_tracker_score_frame_wise = defaultdict(dict)
        self.obj_id_to_last_occluded = {}

        # Hotstart metadata
        self.obj_first_frame_idx = {}
        self.unmatched_frame_inds = defaultdict(list)
        self.overlap_pair_to_frame_inds = defaultdict(list)
        self.trk_keep_alive = {}
        self.removed_obj_ids = set()
        self.suppressed_obj_ids = defaultdict(set)
        self.hotstart_removed_obj_ids = set()

        # Vision feature cache
        self._vision_cache = {}

        # Output buffer (for hotstart)
        self.output_buffer = []

    def obj_id_to_idx(self, obj_id):
        """Get or create index for object ID."""
        if obj_id not in self._obj_id_to_idx:
            idx = len(self._obj_id_to_idx)
            self._obj_id_to_idx[obj_id] = idx
            self._obj_idx_to_id[idx] = obj_id
            if obj_id not in self.obj_ids:
                self.obj_ids.append(obj_id)
            self.max_obj_id = max(self.max_obj_id, obj_id)
        return self._obj_id_to_idx[obj_id]

    def obj_idx_to_id(self, obj_idx):
        return self._obj_idx_to_id[obj_idx]

    def get_obj_num(self):
        return len(self.obj_ids)

    def add_new_frame(self, pixel_values, frame_idx=None):
        """Add a video frame to the session.

        Args:
            pixel_values: (1, H, W, 3) preprocessed frame.
            frame_idx: optional explicit index. Defaults to auto-increment.

        Returns:
            frame_idx used.
        """
        if frame_idx is None:
            frame_idx = self.num_frames
        self.processed_frames[frame_idx] = pixel_values
        self.num_frames = max(self.num_frames, frame_idx + 1)
        return frame_idx

    def add_prompt(self, prompt_text):
        """Add or retrieve prompt ID for a text prompt."""
        for pid, text in self.prompts.items():
            if text == prompt_text:
                return pid
        pid = self._next_prompt_id
        self._next_prompt_id += 1
        self.prompts[pid] = prompt_text
        return pid

    def cache_vision_features(self, frame_idx, features):
        """Cache vision features for a frame."""
        self._vision_cache[frame_idx] = features
        # Evict oldest if over limit
        while len(self._vision_cache) > self.max_vision_cache:
            oldest = min(self._vision_cache.keys())
            del self._vision_cache[oldest]

    def get_cached_vision_features(self, frame_idx):
        return self._vision_cache.get(frame_idx)

    def store_output(self, obj_idx, frame_idx, output, is_conditioning_frame=False):
        """Store model output for an object at a frame."""
        key = (
            "cond_frame_outputs" if is_conditioning_frame else "non_cond_frame_outputs"
        )
        self.output_dict_per_obj[obj_idx][key][frame_idx] = output

    def get_output(self, obj_idx, frame_idx):
        """Retrieve output for an object at a frame (cond first, then non-cond)."""
        d = self.output_dict_per_obj[obj_idx]
        if frame_idx in d["cond_frame_outputs"]:
            return d["cond_frame_outputs"][frame_idx], True
        if frame_idx in d["non_cond_frame_outputs"]:
            return d["non_cond_frame_outputs"][frame_idx], False
        return None, False

    def remove_object(self, obj_id):
        """Remove an object from tracking."""
        if obj_id in self._obj_id_to_idx:
            idx = self._obj_id_to_idx[obj_id]
            del self._obj_id_to_idx[obj_id]
            del self._obj_idx_to_id[idx]
        if obj_id in self.obj_ids:
            self.obj_ids.remove(obj_id)
        self.removed_obj_ids.add(obj_id)

    def reset_tracking_data(self):
        """Clear tracking data but preserve video frames and prompts."""
        self._obj_id_to_idx.clear()
        self._obj_idx_to_id.clear()
        self.obj_ids.clear()
        self.point_inputs_per_obj.clear()
        self.mask_inputs_per_obj.clear()
        self.output_dict_per_obj.clear()
        self.frames_tracked_per_obj.clear()
        self.obj_id_to_score.clear()
        self.obj_id_to_tracker_score_frame_wise.clear()
        self.obj_id_to_last_occluded.clear()
        self.obj_first_frame_idx.clear()
        self.unmatched_frame_inds.clear()
        self.overlap_pair_to_frame_inds.clear()
        self.trk_keep_alive.clear()
        self.removed_obj_ids.clear()
        self.suppressed_obj_ids.clear()
        self.hotstart_removed_obj_ids.clear()
        self.output_buffer.clear()

    def reset_inference_session(self):
        """Full reset: clear everything."""
        self.reset_tracking_data()
        self.processed_frames.clear()
        self._vision_cache.clear()
        self.prompts.clear()
        self.prompt_embeddings.clear()
        self.prompt_attention_masks.clear()
        self.obj_id_to_prompt_id.clear()
        self.num_frames = 0
        self.max_obj_id = 0
        self._next_prompt_id = 0


@keras.saving.register_keras_serializable(package="kmodels")
class Sam3VideoModel(keras.Model):
    """Sam3Video: detector + tracker + vision neck for video understanding.

    Architecture:
        detector_model (Sam3) — runs detection on every frame
        tracker_model (Sam3TrackerVideo) — propagates objects with memory
        tracker_neck (Sam3VisionNeck) — bridges detector features to tracker

    The detector and tracker share the same vision backbone. The tracker_neck
    creates a separate FPN from the backbone output for the tracker format.
    """

    def __init__(
        self,
        sam3_model=None,
        tracker_video_model=None,
        video_config=None,
        **kwargs,
    ):
        super().__init__(name=kwargs.pop("name", "Sam3Video"), **kwargs)

        cfg = video_config or SAM3_VIDEO_MODEL_CONFIG["Sam3Video"]
        self._video_config = cfg

        self.sam3_model = sam3_model
        self.tracker_model = tracker_video_model

        # Vision neck FPN: bridges detector backbone → tracker features
        self.tracker_neck = Sam3VisionNeck(
            backbone_hidden_size=cfg["backbone_hidden_size"],
            fpn_hidden_size=cfg["fpn_hidden_size"],
            scale_factors=cfg["fpn_scale_factors"],
            name="tracker_neck",
        )
        self.tracker_neck.build((None, cfg["backbone_hidden_size"], None, None))

        # Copy config params
        self.score_threshold_detection = cfg["score_threshold_detection"]
        self.det_nms_thresh = cfg["det_nms_thresh"]
        self.new_det_thresh = cfg["new_det_thresh"]
        self.assoc_iou_thresh = cfg["assoc_iou_thresh"]
        self.trk_assoc_iou_thresh = cfg["trk_assoc_iou_thresh"]
        self.recondition_on_trk_masks = cfg["recondition_on_trk_masks"]
        self.recondition_every_nth_frame = cfg["recondition_every_nth_frame"]
        self.high_conf_thresh = cfg["high_conf_thresh"]
        self.high_iou_thresh = cfg["high_iou_thresh"]
        self.hotstart_delay = cfg["hotstart_delay"]
        self.hotstart_unmatch_thresh = cfg["hotstart_unmatch_thresh"]
        self.hotstart_dup_thresh = cfg["hotstart_dup_thresh"]
        self.init_trk_keep_alive = cfg["init_trk_keep_alive"]
        self.max_trk_keep_alive = cfg["max_trk_keep_alive"]
        self.min_trk_keep_alive = cfg["min_trk_keep_alive"]
        self.low_res_mask_size = cfg["low_res_mask_size"]
        self.fill_hole_area = cfg["fill_hole_area"]
        self.max_num_objects = cfg["max_num_objects"]
        self.suppress_occlusion_thresh = cfg[
            "suppress_overlapping_based_on_recent_occlusion_threshold"
        ]
        self.built = True

    def get_backbone_features(self, pixel_values):
        """Extract backbone features from detector's vision encoder.

        Args:
            pixel_values: (B, H, W, 3) NHWC image.

        Returns:
            backbone_nchw: (B, 1024, grid_h, grid_w) backbone output.
        """
        if self.sam3_model is None:
            raise ValueError("sam3_model must be set.")

        from kmodels.models.sam3.sam3_processor import _SUBMODEL_CACHE

        det = self.sam3_model.detector
        cache_key = f"{id(det)}_backbone"
        if cache_key not in _SUBMODEL_CACHE:
            backbone_layer = det.get_layer("backbone_to_nchw")
            _SUBMODEL_CACHE[cache_key] = keras.Model(
                inputs=det.input,
                outputs=backbone_layer.output,
                name="backbone_only",
            )

        backbone_sub = _SUBMODEL_CACHE[cache_key]
        dummy_text = np.zeros((1, 1, 1024), dtype=np.float32)
        dummy_mask = np.ones((1, 1), dtype=np.float32)

        out = backbone_sub.predict(
            {
                "pixel_values": pixel_values,
                "text_features": dummy_text,
                "text_attention_mask": dummy_mask,
            },
            verbose=0,
        )
        return ops.convert_to_tensor(out)

    def get_vision_features_for_tracker(self, backbone_nchw):
        """Bridge detector backbone features to tracker format via FPN neck.

        Args:
            backbone_nchw: (B, 1024, H, W) backbone spatial features.

        Returns:
            feature_maps: list of 3 flattened feature tensors (HW, B, C).
            feature_maps_position_embeddings: list of 3 flattened PE tensors.
        """
        fpn_hidden_states, fpn_position_encoding = self.tracker_neck(backbone_nchw)

        # Use first 3 FPN levels (discard the 0.5x level)
        feature_maps = list(fpn_hidden_states[:3])

        # Pre-project levels 0 and 1 with tracker's mask decoder convs
        feature_maps[0] = self.tracker_model.mask_decoder.conv_s0(feature_maps[0])
        feature_maps[1] = self.tracker_model.mask_decoder.conv_s1(feature_maps[1])

        # Flatten to (HW, B, C) for transformer input
        def flatten_nchw(x):
            # (B, C, H, W) → (B, C, HW) → (HW, B, C)
            s = ops.shape(x)
            x = ops.reshape(x, (s[0], s[1], s[2] * s[3]))
            return ops.transpose(x, (2, 0, 1))

        feature_maps = [flatten_nchw(fm) for fm in feature_maps]
        feature_maps_pe = [flatten_nchw(pe) for pe in fpn_position_encoding[:3]]

        return feature_maps, feature_maps_pe

    def run_detection(self, inference_session, pixel_values, vision_embeds=None):
        """Run text-conditioned detection on a frame.

        Args:
            inference_session: Sam3VideoInferenceSession.
            pixel_values: (1, H, W, 3) preprocessed frame.
            vision_embeds: optional pre-computed backbone features.

        Returns:
            dict[prompt_id → {"bbox": ..., "mask": ..., "scores": ...}]
        """
        if self.sam3_model is None:
            raise ValueError("sam3_model must be set.")

        all_detections = {}
        det = self.sam3_model

        for prompt_id, prompt_text in inference_session.prompts.items():
            # Get or compute text embeddings
            if prompt_id not in inference_session.prompt_embeddings:
                text_encoder = det.text_encoder
                from kmodels.models.sam3.sam3_clip import tokenize

                tokens = tokenize([prompt_text], context_length=77)
                text_feats = text_encoder.predict(tokens, verbose=0)
                inference_session.prompt_embeddings[prompt_id] = text_feats

            text_feats = inference_session.prompt_embeddings[prompt_id]

            # Run detector
            from kmodels.models.sam3.sam3_processor import predict

            result = predict(
                det,
                pixel_values,
                text_feats,
                threshold=self.score_threshold_detection,
            )

            all_detections[prompt_id] = result

        return all_detections

    def run_tracker_propagation(
        self,
        inference_session,
        frame_idx,
        image_embeddings,
        image_pe,
        reverse=False,
    ):
        """Propagate tracked objects through current frame.

        For each tracked object, uses memory-conditioned features
        to predict masks without explicit prompts.

        Args:
            inference_session: Sam3VideoInferenceSession.
            frame_idx: current frame index.
            image_embeddings: list of 3 feature maps from tracker.
            image_pe: (B, 256, H, W) positional embeddings.
            reverse: track in reverse time direction.

        Returns:
            low_res_masks: (num_objects, 1, H_low, W_low)
            obj_scores: (num_objects,)
        """
        num_objs = inference_session.get_obj_num()
        if num_objs == 0:
            return None, None

        all_masks = []
        all_scores = []

        for obj_id in inference_session.obj_ids:
            obj_idx = inference_session.obj_id_to_idx(obj_id)

            # Check if already have conditioning output for this frame
            existing, is_cond = inference_session.get_output(obj_idx, frame_idx)
            if existing is not None and is_cond:
                all_masks.append(existing["pred_masks"])
                all_scores.append(existing.get("object_score_logits", 0.0))
                continue

            # Run single frame with memory conditioning
            output = self._run_single_frame_inference(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idx=obj_idx,
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                reverse=reverse,
            )

            inference_session.store_output(
                obj_idx, frame_idx, output, is_conditioning_frame=False
            )
            all_masks.append(output["pred_masks"])
            all_scores.append(output.get("object_score_logits", 0.0))

        if all_masks:
            low_res_masks = ops.concatenate(all_masks, axis=0)
            obj_scores = ops.convert_to_tensor(all_scores)
            return low_res_masks, obj_scores
        return None, None

    def _run_single_frame_inference(
        self,
        inference_session,
        frame_idx,
        obj_idx,
        image_embeddings,
        image_pe,
        point_inputs=None,
        mask_inputs=None,
        reverse=False,
    ):
        """Run single object inference with optional memory conditioning.

        On initial conditioning frames (with user prompts), runs prompt encoder
        + mask decoder directly. On tracking frames, first conditions features
        with memory attention, then runs mask decoder.
        """
        is_init_cond = (
            obj_idx in inference_session.point_inputs_per_obj
            and frame_idx in inference_session.point_inputs_per_obj[obj_idx]
        ) or (
            obj_idx in inference_session.mask_inputs_per_obj
            and frame_idx in inference_session.mask_inputs_per_obj[obj_idx]
        )

        if point_inputs is None and obj_idx in inference_session.point_inputs_per_obj:
            point_inputs = inference_session.point_inputs_per_obj[obj_idx].get(
                frame_idx
            )
        if mask_inputs is None and obj_idx in inference_session.mask_inputs_per_obj:
            mask_inputs = inference_session.mask_inputs_per_obj[obj_idx].get(frame_idx)

        # Use mask as direct output if provided on conditioning frame
        if is_init_cond and mask_inputs is not None and point_inputs is None:
            return self._use_mask_as_output(mask_inputs, image_embeddings)

        # Memory-condition features if not initial conditioning frame
        if not is_init_cond:
            conditioned_feats = self._prepare_memory_conditioned_features(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_idx=obj_idx,
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                reverse=reverse,
            )
        else:
            # Add no_memory_embedding for initial frames
            fpn_2 = image_embeddings[-1]
            no_mem = ops.reshape(self.tracker_model.no_memory_embedding, (1, -1, 1, 1))
            conditioned_feats = fpn_2 + no_mem

        # Run prompt encoder + mask decoder
        return self._single_frame_forward(
            conditioned_feats=conditioned_feats,
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
        )

    def _use_mask_as_output(self, mask_inputs, image_embeddings):
        """Use provided mask directly as output (bypass SAM decoder)."""
        downsampled = self.tracker_model.mask_downsample(mask_inputs)
        return {
            "pred_masks": downsampled,
            "object_score_logits": ops.zeros((1, 1)),
            "high_res_masks": mask_inputs,
        }

    def _single_frame_forward(
        self,
        conditioned_feats,
        image_embeddings,
        image_pe,
        point_inputs=None,
        mask_inputs=None,
        multimask_output=False,
    ):
        """Run prompt encoder + mask decoder on conditioned features."""
        tracker = self.tracker_model
        batch_size = ops.shape(conditioned_feats)[0]

        if point_inputs is not None:
            input_points = point_inputs.get("point_coords")
            input_labels = point_inputs.get("point_labels")
        else:
            input_points = ops.zeros((batch_size, 1, 1, 2), dtype="float32")
            input_labels = -ops.ones((batch_size, 1, 1), dtype="int32")

        sparse_emb, dense_emb = tracker.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_masks=mask_inputs,
        )

        masks, iou_pred, sam_tokens, object_score = tracker.mask_decoder(
            image_embeddings=conditioned_feats,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:2],
        )

        # Project object pointer
        sam_output_token = sam_tokens[:, :, 0, :]  # First token
        is_appearing = ops.cast(ops.sigmoid(object_score[:, :, 0:1]) > 0.0, "float32")
        obj_pointer = tracker.project_object_pointer(
            sam_output_token[:, 0, :], is_appearing[:, 0, :]
        )

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "object_score_logits": object_score,
            "object_pointer": obj_pointer,
            "sam_tokens": sam_tokens,
        }

    def _prepare_memory_conditioned_features(
        self,
        inference_session,
        frame_idx,
        obj_idx,
        image_embeddings,
        image_pe,
        reverse=False,
    ):
        """Condition current features with temporal memory.

        Gathers memory from past frames and runs memory attention
        to fuse it with current frame features.

        Returns:
            (B, C, H, W) memory-conditioned features.
        """
        tracker = self.tracker_model
        fpn_2 = image_embeddings[-1]  # (B, 256, 72, 72)

        # Gather memory frames
        memory_list = []
        memory_pos_list = []
        obj_pointer_list = []
        temporal_offsets = []

        output_dict = inference_session.output_dict_per_obj.get(obj_idx, {})

        # Collect from conditioning frames
        for fid, out in output_dict.get("cond_frame_outputs", {}).items():
            if "maskmem_features" in out:
                memory_list.append(out["maskmem_features"])
                memory_pos_list.append(out["maskmem_pos_enc"])
            if "object_pointer" in out:
                obj_pointer_list.append(out["object_pointer"])
                temporal_offsets.append(fid - frame_idx)

        # Collect from non-conditioning frames
        for fid, out in output_dict.get("non_cond_frame_outputs", {}).items():
            if "maskmem_features" in out:
                memory_list.append(out["maskmem_features"])
                memory_pos_list.append(out["maskmem_pos_enc"])
            if "object_pointer" in out:
                obj_pointer_list.append(out["object_pointer"])
                temporal_offsets.append(fid - frame_idx)

        if not memory_list:
            # No memory available — use no_memory_embedding
            no_mem = ops.reshape(tracker.no_memory_embedding, (1, -1, 1, 1))
            return fpn_2 + no_mem

        # Stack memories: each (1, mem_dim, H', W') → flatten to (H'W', 1, mem_dim)
        all_mem = []
        all_mem_pos = []
        for mem, mem_pos in zip(memory_list, memory_pos_list):
            s = ops.shape(mem)
            mem_flat = ops.reshape(mem, (s[0], s[1], s[2] * s[3]))
            mem_flat = ops.transpose(mem_flat, (2, 0, 1))  # (HW, B, C)
            all_mem.append(mem_flat)

            s_p = ops.shape(mem_pos)
            pos_flat = ops.reshape(mem_pos, (s_p[0], s_p[1], s_p[2] * s_p[3]))
            pos_flat = ops.transpose(pos_flat, (2, 0, 1))
            all_mem_pos.append(pos_flat)

        memory = ops.concatenate(all_mem, axis=0)  # (total_HW, B, mem_dim)
        memory_pos = ops.concatenate(all_mem_pos, axis=0)

        # Process object pointers
        num_ptr_tokens = 0
        if obj_pointer_list:
            ptrs = ops.stack(obj_pointer_list, axis=0)  # (N, hidden_dim)
            ptrs = ops.expand_dims(ptrs, 1)  # (N, 1, hidden_dim)

            # Temporal PE for pointers
            if self.tracker_model._enable_temporal_pos and temporal_offsets:
                max_offset = max(abs(t) for t in temporal_offsets) or 1
                norm_offsets = ops.convert_to_tensor(
                    [t / max_offset for t in temporal_offsets], dtype="float32"
                )
                from kmodels.models.sam3_tracker_video.sam3_tracker_video_layers import (
                    get_1d_sine_pe,
                )

                sine_pe = get_1d_sine_pe(
                    norm_offsets, dim=self.tracker_model._hidden_dim
                )
                proj_pe = tracker.temporal_positional_encoding_projection_layer(sine_pe)
                proj_pe = ops.expand_dims(proj_pe, 1)  # (N, 1, mem_dim)
            else:
                proj_pe = ops.zeros_like(ptrs[..., : self.tracker_model._mem_dim])

            # Reshape pointers to mem_dim if needed
            if self.tracker_model._hidden_dim != self.tracker_model._mem_dim:
                ratio = self.tracker_model._hidden_dim // self.tracker_model._mem_dim
                ptrs_reshaped = ops.reshape(
                    ptrs,
                    (
                        ops.shape(ptrs)[0],
                        ops.shape(ptrs)[1] * ratio,
                        self.tracker_model._mem_dim,
                    ),
                )
                proj_pe = ops.repeat(proj_pe, ratio, axis=1)
            else:
                ptrs_reshaped = ptrs

            num_ptr_tokens = ops.shape(ptrs_reshaped)[0] * ops.shape(ptrs_reshaped)[1]

            # Flatten pointers and append to memory
            # ptrs: (N, B_ratio, mem_dim) → (N*B_ratio, 1, mem_dim)
            s_ptr = ops.shape(ptrs_reshaped)
            ptr_flat = ops.reshape(ptrs_reshaped, (s_ptr[0] * s_ptr[1], 1, s_ptr[2]))
            pe_flat = ops.reshape(proj_pe, (s_ptr[0] * s_ptr[1], 1, s_ptr[2]))

            memory = ops.concatenate([memory, ptr_flat], axis=0)
            memory_pos = ops.concatenate([memory_pos, pe_flat], axis=0)

        # Current vision features: (B, C, H, W) → (HW, B, C)
        s_f = ops.shape(fpn_2)
        current_feats = ops.reshape(fpn_2, (s_f[0], s_f[1], s_f[2] * s_f[3]))
        current_feats = ops.transpose(current_feats, (2, 0, 1))

        # Current position embeddings
        current_pe = ops.reshape(image_pe, (s_f[0], s_f[1], s_f[2] * s_f[3]))
        current_pe = ops.transpose(current_pe, (2, 0, 1))

        # Run memory attention
        conditioned = tracker.memory_attention(
            current_vision_features=current_feats,
            memory=memory,
            current_vision_position_embeddings=current_pe,
            memory_position_embeddings=memory_pos,
            num_object_pointer_tokens=num_ptr_tokens,
        )

        # Reshape back: (HW, B, C) → (B, C, H, W)
        conditioned = ops.transpose(conditioned, (1, 2, 0))
        conditioned = ops.reshape(conditioned, (s_f[0], s_f[1], s_f[2], s_f[3]))

        return conditioned

    def encode_and_store_memory(
        self,
        inference_session,
        frame_idx,
        obj_idx,
        vision_features,
        high_res_masks,
        object_score_logits=None,
        is_conditioning_frame=False,
    ):
        """Encode masks into memory and store in inference session.

        Args:
            vision_features: (B, C, H, W) FPN level 2 features.
            high_res_masks: (B, 1, H_mask, W_mask) high-res masks.
        """
        maskmem_features, maskmem_pos_enc = self.tracker_model.encode_memory(
            vision_features, high_res_masks
        )

        output, _ = inference_session.get_output(obj_idx, frame_idx)
        if output is None:
            output = {}

        output["maskmem_features"] = maskmem_features
        output["maskmem_pos_enc"] = maskmem_pos_enc

        inference_session.store_output(
            obj_idx, frame_idx, output, is_conditioning_frame
        )

    @staticmethod
    def mask_iou(pred_masks, gt_masks):
        """Compute mask IoU between two sets of binary masks.

        Args:
            pred_masks: (N, H, W) predicted masks.
            gt_masks: (M, H, W) ground truth masks.

        Returns:
            (N, M) IoU matrix.
        """
        pred_flat = ops.reshape(pred_masks, (ops.shape(pred_masks)[0], -1))
        gt_flat = ops.reshape(gt_masks, (ops.shape(gt_masks)[0], -1))

        pred_flat = ops.cast(pred_flat > 0, "float32")
        gt_flat = ops.cast(gt_flat > 0, "float32")

        intersection = ops.matmul(pred_flat, ops.transpose(gt_flat, (1, 0)))
        pred_areas = ops.sum(pred_flat, axis=-1, keepdims=True)
        gt_areas = ops.sum(gt_flat, axis=-1, keepdims=True)
        union = pred_areas + ops.transpose(gt_areas, (1, 0)) - intersection

        return intersection / ops.maximum(union, 1e-6)

    def call(
        self,
        inference_session,
        frame_idx=None,
        frame=None,
        reverse=False,
    ):
        """Process a single video frame: detect + track + associate.

        Args:
            inference_session: Sam3VideoInferenceSession.
            frame_idx: which frame to process.
            frame: (1, H, W, 3) new frame to add (optional).
            reverse: track in reverse direction.

        Returns:
            dict with object_ids, obj_id_to_mask, obj_id_to_score, frame_idx.
        """
        if frame is not None:
            frame_idx = inference_session.add_new_frame(frame, frame_idx)

        if frame_idx is None:
            raise ValueError("Must provide frame_idx or frame.")

        pixel_values = inference_session.processed_frames[frame_idx]

        # 1. Extract backbone features
        backbone_nchw = self.get_backbone_features(pixel_values)

        # 2. Run detection
        det_out = self.run_detection(inference_session, pixel_values)

        # 3. Bridge to tracker via FPN neck
        feature_maps, feature_maps_pe = self.get_vision_features_for_tracker(
            backbone_nchw
        )

        # Prepare tracker-format embeddings
        # feature_maps[-1] is the 1x level: (HW, B, C)
        # Reshape to (B, C, H, W) for tracker
        grid_size = self.tracker_model._image_embedding_size
        fpn_2_flat = feature_maps[-1]  # (HW, B, C)
        fpn_2_spatial = ops.transpose(fpn_2_flat, (1, 2, 0))  # (B, C, HW)
        fpn_2_spatial = ops.reshape(
            fpn_2_spatial,
            (
                ops.shape(fpn_2_spatial)[0],
                ops.shape(fpn_2_spatial)[1],
                grid_size,
                grid_size,
            ),
        )

        image_embeddings = [feature_maps[0], feature_maps[1], fpn_2_spatial]
        image_pe = self.tracker_model.get_image_wide_positional_embeddings()
        image_pe = ops.broadcast_to(
            image_pe, (ops.shape(fpn_2_spatial)[0],) + ops.shape(image_pe)[1:]
        )

        # 4. Propagate tracker
        low_res_masks, obj_scores = self.run_tracker_propagation(
            inference_session, frame_idx, image_embeddings, image_pe, reverse
        )

        # 5. Build output
        obj_id_to_mask = {}
        obj_id_to_score = {}

        if low_res_masks is not None:
            for i, obj_id in enumerate(inference_session.obj_ids):
                obj_id_to_mask[obj_id] = low_res_masks[i : i + 1]
                if obj_scores is not None:
                    obj_id_to_score[obj_id] = float(ops.convert_to_numpy(obj_scores[i]))

        # Add new detections
        for prompt_id, det_result in det_out.items():
            if "masks" in det_result and det_result["masks"] is not None:
                for j in range(len(det_result.get("scores", []))):
                    new_id = inference_session.max_obj_id + 1
                    inference_session.max_obj_id = new_id
                    inference_session.obj_id_to_prompt_id[new_id] = prompt_id
                    obj_id_to_mask[new_id] = det_result["masks"][j : j + 1]
                    obj_id_to_score[new_id] = float(det_result["scores"][j])

        return {
            "object_ids": list(obj_id_to_mask.keys()),
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "frame_idx": frame_idx,
        }

    def propagate_in_video_iterator(
        self,
        inference_session,
        start_frame_idx=0,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Iterate through video frames and yield per-frame results.

        Args:
            inference_session: Sam3VideoInferenceSession.
            start_frame_idx: first frame to process.
            max_frame_num_to_track: max frames to process.
            reverse: process frames in reverse order.

        Yields:
            dict per frame with object_ids, obj_id_to_mask, obj_id_to_score.
        """
        frame_indices = sorted(inference_session.processed_frames.keys())
        if reverse:
            frame_indices = frame_indices[::-1]

        if start_frame_idx in frame_indices:
            start_pos = frame_indices.index(start_frame_idx)
            frame_indices = frame_indices[start_pos:]

        if max_frame_num_to_track is not None:
            frame_indices = frame_indices[:max_frame_num_to_track]

        for fidx in frame_indices:
            result = self(inference_session, frame_idx=fidx, reverse=reverse)
            yield result

    def get_config(self):
        config = super().get_config()
        config.update({"video_config": self._video_config})
        return config


def _create_sam3_video(
    variant, sam3_model=None, tracker_video_model=None, weights=None, **kwargs
):
    config = SAM3_VIDEO_MODEL_CONFIG[variant]
    model = Sam3VideoModel(
        sam3_model=sam3_model,
        tracker_video_model=tracker_video_model,
        video_config=config,
        **kwargs,
    )

    valid_weights = list(SAM3_VIDEO_WEIGHTS_CONFIG.get(variant, {}).keys())
    if weights in valid_weights:
        from kmodels.models.sam3.weights_config import load_unified_weights

        load_unified_weights(
            sam3_model=sam3_model,
            tracker_video_model=tracker_video_model,
            video_model=model,
            weights=weights,
        )
    elif weights is not None:
        model.load_weights(weights, skip_mismatch=True)
    else:
        print("No video model weights loaded.")

    return model


@register_model
def Sam3Video(sam3_model=None, tracker_video_model=None, weights=None, **kwargs):
    """Create a Sam3Video model.

    Args:
        sam3_model: a trained SAM3Model (from Sam3(weights="pcs")).
        tracker_video_model: a Sam3TrackerVideoModel.
        weights: weight variant name or file path.

    Usage:
        from kmodels.models.sam3 import Sam3
        from kmodels.models.sam3_tracker_video import Sam3TrackerVideo
        from kmodels.models.sam3_video import Sam3Video

        sam3 = Sam3(weights="pcs")
        tracker_video = Sam3TrackerVideo(sam3_model=sam3, weights="pcs")
        video = Sam3Video(
            sam3_model=sam3,
            tracker_video_model=tracker_video,
            weights="pcs",
        )
    """
    return _create_sam3_video(
        "Sam3Video",
        sam3_model=sam3_model,
        tracker_video_model=tracker_video_model,
        weights=weights,
        **kwargs,
    )
