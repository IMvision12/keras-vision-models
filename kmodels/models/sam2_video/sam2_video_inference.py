from collections import OrderedDict

import keras
import numpy as np
from keras import ops

from kmodels.models.sam2_video.sam2_video_model import (
    Sam2Video,
    sam2_video_feed_forward_call,
    sam2_video_memory_encoder_call,
    sam2_video_sine_position_embedding,
)

NUM_MASKMEM = 7
MEM_DIM = 64
MAX_OBJECT_POINTERS_IN_ENCODER = 16
SIGMOID_SCALE_FOR_MEM_ENC = 20.0
SIGMOID_BIAS_FOR_MEM_ENC = -10.0


class Sam2VideoInferenceSession:
    def __init__(self, processed_frames, video_height, video_width):
        self.processed_frames = processed_frames
        self.video_height = video_height
        self.video_width = video_width
        self.num_frames = len(processed_frames)

        self._obj_id_to_idx = OrderedDict()
        self._obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        self.point_inputs_per_obj = {}

        self.output_dict_per_obj = {}

        self.frames_tracked_per_obj = {}

    def obj_id_to_idx(self, obj_id):
        if obj_id in self._obj_id_to_idx:
            return self._obj_id_to_idx[obj_id]
        idx = len(self._obj_id_to_idx)
        self._obj_id_to_idx[obj_id] = idx
        self._obj_idx_to_id[idx] = obj_id
        self.obj_ids.append(obj_id)
        self.point_inputs_per_obj[idx] = {}
        self.output_dict_per_obj[idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        self.frames_tracked_per_obj[idx] = {}
        return idx

    def add_point_inputs(self, obj_id, frame_idx, point_coords, point_labels):
        obj_idx = self.obj_id_to_idx(obj_id)
        self.point_inputs_per_obj[obj_idx][frame_idx] = {
            "point_coords": np.asarray(point_coords, dtype=np.float32),
            "point_labels": np.asarray(point_labels, dtype=np.int32),
        }

    def store_output(self, obj_idx, frame_idx, output, is_conditioning):
        key = "cond_frame_outputs" if is_conditioning else "non_cond_frame_outputs"
        self.output_dict_per_obj[obj_idx][key][frame_idx] = output

    def get_num_objects(self):
        return len(self._obj_id_to_idx)


def _build_sub_models(sam2_video_model):
    pixel_values_input = sam2_video_model.get_layer("pixel_values").output
    input_points_input = sam2_video_model.get_layer("input_points").output
    input_labels_input = sam2_video_model.get_layer("input_labels").output

    no_mem_embed_layer = sam2_video_model.get_layer("no_memory_embedding")
    prompt_enc_layer = sam2_video_model.get_layer("prompt_encoder")
    mask_dec_layer = sam2_video_model.get_layer("mask_decoder")
    image_pe_layer = sam2_video_model.get_layer("image_positional_embeddings")

    image_embeddings_raw = no_mem_embed_layer.input
    image_embeddings_post_no_mem = no_mem_embed_layer.output
    high_res_feat_s0 = sam2_video_model.get_layer("neck_convs_3").output
    high_res_feat_s1 = sam2_video_model.get_layer("neck_convs_2").output
    image_pe = image_pe_layer.output

    encoder_sub = keras.Model(
        inputs=pixel_values_input,
        outputs={
            "image_embeddings_raw": image_embeddings_raw,
            "image_embeddings_post_no_mem": image_embeddings_post_no_mem,
            "high_res_feat_s0": high_res_feat_s0,
            "high_res_feat_s1": high_res_feat_s1,
            "image_pe": image_pe,
        },
        name="encoder_sub",
    )

    prompt_out = prompt_enc_layer([input_points_input, input_labels_input])
    prompt_sub = keras.Model(
        inputs=[input_points_input, input_labels_input],
        outputs={
            "sparse_embeddings": prompt_out["sparse_embeddings"],
            "dense_embeddings": prompt_out["dense_embeddings"],
        },
        name="prompt_sub",
    )

    return encoder_sub, prompt_sub, mask_dec_layer, no_mem_embed_layer


def _gather_memory_frames(session, obj_idx, frame_idx, num_maskmem=NUM_MASKMEM):
    obj_out = session.output_dict_per_obj[obj_idx]
    cond_outputs = obj_out["cond_frame_outputs"]
    non_cond_outputs = obj_out["non_cond_frame_outputs"]

    memory_frames = []

    for cond_frame_idx, cond_out in cond_outputs.items():
        memory_frames.append((0, cond_out))

    for k in range(num_maskmem - 1, 0, -1):
        prev_idx = frame_idx - k
        if prev_idx in non_cond_outputs:
            memory_frames.append((k, non_cond_outputs[prev_idx]))

    memory_frames.sort(key=lambda x: x[0])
    return memory_frames


def _gather_object_pointers(
    session, obj_idx, frame_idx, max_ptrs=MAX_OBJECT_POINTERS_IN_ENCODER
):
    obj_out = session.output_dict_per_obj[obj_idx]
    cond_outputs = obj_out["cond_frame_outputs"]
    non_cond_outputs = obj_out["non_cond_frame_outputs"]

    cond_ptrs = []
    for cf_idx, cf_out in cond_outputs.items():
        if cf_idx < frame_idx and "object_pointer" in cf_out:
            t_diff = abs(frame_idx - cf_idx)
            cond_ptrs.append((t_diff, cf_out["object_pointer"]))

    remaining = max_ptrs - len(cond_ptrs)
    non_cond_ptrs = []
    for k in range(1, remaining + 1):
        prev_idx = frame_idx - k
        if (
            prev_idx in non_cond_outputs
            and "object_pointer" in non_cond_outputs[prev_idx]
        ):
            non_cond_ptrs.append((k, non_cond_outputs[prev_idx]["object_pointer"]))

    all_ptrs = cond_ptrs + non_cond_ptrs
    all_ptrs.sort(key=lambda x: x[0])
    return all_ptrs


def _sine_pos_encoding_1d(dim, t_array):
    pe_dim = dim // 2
    i_range = np.arange(pe_dim, dtype=np.float32)
    dim_t = 10000.0 ** (2 * (i_range // 2) / pe_dim)
    angles = t_array[:, None].astype(np.float32) / dim_t[None, :]
    pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    return pe


def _prepare_memory_conditioned_features(
    sam2_video_model,
    session,
    obj_idx,
    frame_idx,
    current_vision_feats_channels_first,
    current_vision_pos_embeds,
):
    mem_frames = _gather_memory_frames(session, obj_idx, frame_idx)

    obj_ptrs = _gather_object_pointers(session, obj_idx, frame_idx)

    if len(mem_frames) == 0 and len(obj_ptrs) == 0:
        return ops.transpose(current_vision_feats_channels_first, [0, 2, 3, 1])

    B, C, H, W = ops.shape(current_vision_feats_channels_first)
    current_feats_bhwc = ops.transpose(
        current_vision_feats_channels_first, [0, 2, 3, 1]
    )
    current_feats_flat = ops.reshape(current_feats_bhwc, (1, H * W, C))

    pos_bhwc = ops.transpose(current_vision_pos_embeds, [0, 2, 3, 1])
    pos_flat = ops.reshape(pos_bhwc, (1, H * W, C))

    memory_feats_list = []
    memory_pos_list = []

    mem_temporal_pe = sam2_video_model.memory_temporal_positional_encoding.numpy()

    for temporal_offset, mem_out in mem_frames:
        maskmem = mem_out["maskmem_features"]
        maskmem_pos = mem_out["maskmem_pos_enc"]

        mm = ops.convert_to_tensor(maskmem)
        mm_pos = ops.convert_to_tensor(maskmem_pos)
        mm = ops.transpose(mm, [0, 2, 3, 1])
        mm_pos = ops.transpose(mm_pos, [0, 2, 3, 1])

        h_m = ops.shape(mm)[1]
        w_m = ops.shape(mm)[2]
        mm = ops.reshape(mm, (1, h_m * w_m, MEM_DIM))
        mm_pos = ops.reshape(mm_pos, (1, h_m * w_m, MEM_DIM))

        tpe_idx = temporal_offset - 1
        tpe = mem_temporal_pe[tpe_idx]
        tpe_tensor = ops.convert_to_tensor(tpe.reshape(1, 1, MEM_DIM))
        mm_pos = mm_pos + tpe_tensor

        memory_feats_list.append(mm)
        memory_pos_list.append(mm_pos)

    num_obj_ptr_tokens = 0
    if len(obj_ptrs) > 0:
        t_diffs = np.array([t for t, _ in obj_ptrs], dtype=np.float32)
        t_diffs_normalized = t_diffs / (MAX_OBJECT_POINTERS_IN_ENCODER - 1)

        pointer_tensors = [ops.convert_to_tensor(p) for _, p in obj_ptrs]
        pointers_stacked = ops.concatenate(pointer_tensors, axis=1)

        N = len(obj_ptrs)
        pointers_4x = ops.reshape(pointers_stacked, (1, N * 4, MEM_DIM))

        pe_1d = _sine_pos_encoding_1d(256, t_diffs_normalized)
        pe_1d_tensor = ops.convert_to_tensor(pe_1d[None, :, :])
        pe_1d_proj = sam2_video_model.temporal_pos_enc_proj(pe_1d_tensor)

        pe_1d_proj_4x = ops.repeat(pe_1d_proj, 4, axis=1)

        memory_feats_list.append(pointers_4x)
        memory_pos_list.append(pe_1d_proj_4x)
        num_obj_ptr_tokens = N * 4

    combined_memory = ops.concatenate(memory_feats_list, axis=1)
    combined_memory_pos = ops.concatenate(memory_pos_list, axis=1)

    output = sam2_video_model.memory_attention(
        current_vision_feats=current_feats_flat,
        memory=combined_memory,
        current_vision_pos_embeds=pos_flat,
        memory_pos_embeds=combined_memory_pos,
        num_object_pointer_tokens=num_obj_ptr_tokens,
        training=False,
    )

    output_bhwc = ops.reshape(output, (1, H, W, 256))
    return output_bhwc


def _encode_memory(
    sam2_video_model,
    vision_features_bhwc,
    high_res_mask_bhwc,
    is_mask_from_pts=False,
):
    if is_mask_from_pts:
        mask_for_mem_bhwc = ops.cast(high_res_mask_bhwc > 0, "float32")
    else:
        mask_for_mem_bhwc = ops.sigmoid(high_res_mask_bhwc)
    mask_for_mem_bhwc = (
        mask_for_mem_bhwc * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC
    )
    vf_bchw = ops.transpose(vision_features_bhwc, [0, 3, 1, 2])
    mask_bchw = ops.transpose(mask_for_mem_bhwc, [0, 3, 1, 2])

    maskmem_features_bchw, maskmem_pos_enc_bchw = sam2_video_memory_encoder_call(
        vision_features=vf_bchw,
        masks=mask_bchw,
        ds_convs=sam2_video_model.mem_enc_ds_convs,
        ds_lns=sam2_video_model.mem_enc_ds_lns,
        ds_final_conv=sam2_video_model.mem_enc_ds_final_conv,
        feature_proj=sam2_video_model.mem_enc_feature_proj,
        fuser_dw_convs=sam2_video_model.mem_enc_fuser_dw_convs,
        fuser_lns=sam2_video_model.mem_enc_fuser_lns,
        fuser_pw1s=sam2_video_model.mem_enc_fuser_pw1s,
        fuser_pw2s=sam2_video_model.mem_enc_fuser_pw2s,
        fuser_scales=sam2_video_model.mem_enc_fuser_scales,
        projection=sam2_video_model.mem_enc_projection,
        num_pos_feats=MEM_DIM // 2,
    )
    return maskmem_features_bchw, maskmem_pos_enc_bchw


def _compute_object_pointer(
    sam2_video_model, mask_tokens_out_all, iou_scores_all, use_best_mask=True
):
    if use_best_mask:
        iou = iou_scores_all[:, :, 1:]
        tokens = mask_tokens_out_all[:, :, 1:, :]
        best_idx = ops.argmax(iou, axis=-1)
        best_token = ops.take_along_axis(
            tokens, ops.expand_dims(best_idx, axis=-1)[..., None], axis=2
        )[:, :, 0, :]
    else:
        best_token = mask_tokens_out_all[:, :, 0, :]

    best_token_flat = ops.reshape(best_token, (1, 256))
    object_pointer = sam2_video_feed_forward_call(
        best_token_flat,
        sam2_video_model.obj_ptr_proj_in,
        sam2_video_model.obj_ptr_proj_hidden_layers,
        sam2_video_model.obj_ptr_proj_out,
    )
    return ops.reshape(object_pointer, (1, 1, 256))


class Sam2VideoPredictor:
    def __init__(self, sam2_video_model):
        assert isinstance(sam2_video_model, Sam2Video)
        self.model = sam2_video_model
        (
            self.encoder_sub,
            self.prompt_sub,
            self.decoder_layer,
            self.no_mem_embed_layer,
        ) = _build_sub_models(sam2_video_model)

    def _encode_frame(self, frame_bhwc):
        return self.encoder_sub(frame_bhwc, training=False)

    def _encode_prompt(self, points, labels):
        return self.prompt_sub(
            [ops.convert_to_tensor(points), ops.convert_to_tensor(labels)],
            training=False,
        )

    def _run_decoder(
        self,
        image_embeddings,
        image_pe,
        sparse_embeddings,
        dense_embeddings,
        high_res_feat_s0,
        high_res_feat_s1,
    ):
        return self.decoder_layer(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
                high_res_feat_s0,
                high_res_feat_s1,
            ]
        )

    def run_frame(
        self,
        session,
        frame_idx,
        is_init_cond_frame=False,
    ):
        frame = session.processed_frames[frame_idx]
        enc = self._encode_frame(ops.convert_to_tensor(frame))

        image_embeddings_raw = enc["image_embeddings_raw"]
        image_pe = enc["image_pe"]
        high_res_feat_s0 = enc["high_res_feat_s0"]
        high_res_feat_s1 = enc["high_res_feat_s1"]

        image_emb_raw_bchw = ops.transpose(image_embeddings_raw, [0, 3, 1, 2])
        sine_pe_bchw = sam2_video_sine_position_embedding(
            image_emb_raw_bchw, num_pos_feats=128
        )

        results = {}
        num_objs = session.get_num_objects()

        for obj_idx in range(num_objs):
            has_cond = (
                frame_idx in session.output_dict_per_obj[obj_idx]["cond_frame_outputs"]
            )
            has_prompt = frame_idx in session.point_inputs_per_obj[obj_idx]
            is_cond = is_init_cond_frame or has_prompt

            if is_cond and not has_cond:
                image_embeddings = self.no_mem_embed_layer(image_embeddings_raw)
            else:
                conditioned_bhwc = _prepare_memory_conditioned_features(
                    self.model,
                    session,
                    obj_idx,
                    frame_idx,
                    image_emb_raw_bchw,
                    sine_pe_bchw,
                )
                image_embeddings = conditioned_bhwc

            if has_prompt:
                pt_data = session.point_inputs_per_obj[obj_idx][frame_idx]
                points = pt_data["point_coords"]
                labels = pt_data["point_labels"]
                points = np.asarray(points, dtype=np.float32)
                labels = np.asarray(labels, dtype=np.int32)
                if points.ndim == 2:
                    points = points[None, None, :, :]
                elif points.ndim == 3:
                    points = points[None, :, :, :]
                if labels.ndim == 1:
                    labels = labels[None, None, :]
                elif labels.ndim == 2:
                    labels = labels[None, :, :]
            else:
                points = np.zeros((1, 1, 1, 2), dtype=np.float32)
                labels = -np.ones((1, 1, 1), dtype=np.int32)

            prompt_out = self._encode_prompt(points, labels)
            sparse = prompt_out["sparse_embeddings"]
            dense = prompt_out["dense_embeddings"]

            dec_out = self._run_decoder(
                image_embeddings,
                image_pe,
                sparse,
                dense,
                high_res_feat_s0,
                high_res_feat_s1,
            )

            pred_masks_all = dec_out["pred_masks"]
            iou_scores_all = dec_out["iou_scores"]
            object_score_logits = dec_out["object_score_logits"]
            mask_tokens_out_all = dec_out["mask_tokens_out"]

            iou = iou_scores_all[:, :, 1:]
            best_idx = ops.argmax(iou, axis=-1)
            best_idx_scalar = int(ops.convert_to_numpy(best_idx).flatten()[0])
            pred_masks_best = pred_masks_all[
                :, :, 1 + best_idx_scalar : 2 + best_idx_scalar, :, :
            ]

            import torch.nn.functional as F

            pm = pred_masks_best[0, 0]
            pm_bchw = ops.expand_dims(pm, 0)
            pm_bchw_up = F.interpolate(
                pm_bchw,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            high_res_mask_bhwc = ops.transpose(pm_bchw_up, [0, 2, 3, 1])

            obj_ptr = _compute_object_pointer(
                self.model, mask_tokens_out_all, iou_scores_all, use_best_mask=True
            )

            maskmem_features_bchw, maskmem_pos_enc_bchw = _encode_memory(
                self.model,
                image_embeddings_raw,
                high_res_mask_bhwc,
                is_mask_from_pts=has_prompt,
            )

            frame_output = {
                "pred_masks": ops.convert_to_numpy(pred_masks_best),
                "high_res_masks": ops.convert_to_numpy(high_res_mask_bhwc),
                "object_pointer": ops.convert_to_numpy(obj_ptr),
                "object_score_logits": ops.convert_to_numpy(object_score_logits),
                "maskmem_features": ops.convert_to_numpy(maskmem_features_bchw),
                "maskmem_pos_enc": ops.convert_to_numpy(maskmem_pos_enc_bchw),
            }
            session.store_output(
                obj_idx, frame_idx, frame_output, is_conditioning=is_cond
            )

            if not is_cond:
                session.frames_tracked_per_obj[obj_idx][frame_idx] = True

            results[obj_idx] = frame_output

        return results

    def propagate_in_video(self, session, start_frame_idx=0):
        for frame_idx in range(start_frame_idx, session.num_frames):
            any_prompt = any(
                frame_idx in session.point_inputs_per_obj.get(oi, {})
                for oi in range(session.get_num_objects())
            )
            is_init_cond = any_prompt or (
                frame_idx == start_frame_idx
                and all(
                    not session.output_dict_per_obj[oi]["cond_frame_outputs"]
                    for oi in range(session.get_num_objects())
                )
            )
            results = self.run_frame(
                session, frame_idx, is_init_cond_frame=is_init_cond
            )
            yield frame_idx, results
