"""Multimodal assembly + greedy generation for Qwen2-VL.

Glues together:
  * ``Qwen2VLImageProcessor``  -> pixel_values + image_grid_thw
  * ``Qwen2VLTokenizer``       -> input_ids (with ``<|image_pad|>`` placeholders)
  * vision encoder             -> vision_embeds (N_merged_patches, hidden)
  * embed_tokens               -> text_embeds (B, T, hidden)
  * scatter vision_embeds into text_embeds at ``<|image_pad|>`` positions
  * build per-axis M-RoPE position_ids (HF ``get_rope_index`` semantics)
  * LLM forward
"""

import numpy as np
from keras import ops

from .qwen2_vl_layers import build_mrope_cos_sin

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652


def build_multimodal_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    spatial_merge_size: int = 2,
    image_token_id: int = IMAGE_TOKEN_ID,
    video_token_id: int = VIDEO_TOKEN_ID,
    vision_start_token_id: int = VISION_START_TOKEN_ID,
) -> np.ndarray:
    """Port of HF ``get_rope_index`` (image-only case).

    Returns position_ids of shape ``(3, B, T)``: rows are temporal / height /
    width indices. For text-only tokens all three rows share the same value.
    For image tokens the rows diverge and reflect the 3D grid layout.
    """
    B, T = input_ids.shape
    position_ids = np.zeros((3, B, T), dtype=np.int64)

    for b in range(B):
        ids = input_ids[b]
        vision_starts = np.where(ids == vision_start_token_id)[0]
        image_nums = (ids[vision_starts + 1] == image_token_id).sum()
        video_nums = (ids[vision_starts + 1] == video_token_id).sum()

        img_idx = 0  # index into image_grid_thw
        vid_idx = 0
        remain_images = int(image_nums)
        remain_videos = int(video_nums)

        st = 0  # text cursor
        st_cursor = 0  # next free position (max so far + 1)
        llm_pos_ids_list = []

        while st < T:
            ed_image = T + 1
            ed_video = T + 1
            if remain_images > 0:
                ed_image = int(np.where(ids[st:] == image_token_id)[0][0] + st)
            if remain_videos > 0:
                ed_video = int(np.where(ids[st:] == video_token_id)[0][0] + st)
            if ed_image < ed_video:
                t_len, h_len, w_len = [int(x) for x in image_grid_thw[img_idx]]
                img_idx += 1
                remain_images -= 1
                ed = ed_image
            elif ed_video < ed_image:
                t_len, h_len, w_len = [int(x) for x in image_grid_thw[vid_idx]]
                vid_idx += 1
                remain_videos -= 1
                ed = ed_video
            else:
                # no more vision tokens
                text_len = T - st
                llm_pos_ids_list.append(
                    np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_cursor
                )
                break

            # text before vision
            text_len = ed - st
            if text_len > 0:
                llm_pos_ids_list.append(
                    np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_cursor
                )
                st_cursor += text_len

            # vision tokens: (t_len * h_len/ms * w_len/ms) merged patches
            h = h_len // spatial_merge_size
            w = w_len // spatial_merge_size
            t = t_len
            t_idx = np.arange(t).reshape(-1, 1).repeat(h * w, axis=1).flatten()
            h_idx = (
                np.arange(h)
                .reshape(1, -1, 1)
                .repeat(t, axis=0)
                .repeat(w, axis=2)
                .flatten()
            )
            w_idx = (
                np.arange(w)
                .reshape(1, 1, -1)
                .repeat(t, axis=0)
                .repeat(h, axis=1)
                .flatten()
            )
            vision_pos = np.stack([t_idx, h_idx, w_idx], axis=0) + st_cursor
            llm_pos_ids_list.append(vision_pos)
            st_cursor += int(max(t, h, w))
            st = ed + t * h * w

        pos_ids = np.concatenate(llm_pos_ids_list, axis=1)[:, :T]
        position_ids[:, b, : pos_ids.shape[1]] = pos_ids

    return position_ids.astype(np.int32)


def scatter_vision_into_embeds(
    input_ids: np.ndarray,
    text_embeds: np.ndarray,
    vision_embeds: np.ndarray,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> np.ndarray:
    """Replace embeddings at ``image_token_id`` positions with vision features.

    ``text_embeds`` shape: ``(B, T, C)``. ``vision_embeds`` shape:
    ``(N_merged_total, C)`` — total merged patches across all images, in
    order of appearance in ``input_ids``.
    """
    out = text_embeds.copy()
    B, T, C = text_embeds.shape
    v_idx = 0
    for b in range(B):
        mask = input_ids[b] == image_token_id
        n_img_tokens = int(mask.sum())
        out[b, mask] = vision_embeds[v_idx : v_idx + n_img_tokens]
        v_idx += n_img_tokens
    return out


def make_full_causal_mask(seq_len: int) -> np.ndarray:
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    return ((j > i).astype(np.float32) * -1e9)[None, None, :, :]


def qwen2_vl_encode_inputs(
    bundle,
    prompt_text: str,
    tokenizer,
    image_processor,
    images,
) -> dict:
    """Prepare everything the LLM needs from a prompt + images."""
    from .qwen2_vl_layers import build_vision_rope_cos_sin

    cfg = bundle["config"]
    vc = cfg["vision_config"]
    ms = vc["spatial_merge_size"]

    img_inputs = (
        image_processor(images)
        if images
        else {
            "pixel_values": np.zeros((0, 1176), dtype=np.float32),
            "image_grid_thw": np.zeros((0, 3), dtype=np.int32),
        }
    )
    grid_thw = img_inputs["image_grid_thw"]

    # Count merged patches per image -> build prompt with placeholders.
    merged_counts = [int(t) * int(h) // ms * int(w) // ms for t, h, w in grid_thw]
    prompt = tokenizer.build_chat_prompt(prompt_text, image_patch_counts=merged_counts)
    input_ids = np.asarray(
        tokenizer.tokenize(prompt, add_special_tokens=False), dtype=np.int32
    )
    input_ids = input_ids[None, :]  # (1, T)

    # Run vision if images provided.
    head_dim = vc["embed_dim"] // vc["num_heads"]
    if grid_thw.shape[0] > 0:
        cos_v, sin_v = build_vision_rope_cos_sin(
            grid_thw, bundle["vision_inv_freq"], head_dim, ms
        )
        vision_out = bundle["vision"](
            {
                "pixel_values": img_inputs["pixel_values"],
                "vision_cos": ops.convert_to_numpy(cos_v),
                "vision_sin": ops.convert_to_numpy(sin_v),
            }
        )
        vision_embeds = ops.convert_to_numpy(vision_out)
    else:
        vision_embeds = np.zeros(
            (0, cfg["text_config"]["hidden_size"]), dtype=np.float32
        )

    text_embeds = ops.convert_to_numpy(
        bundle["embed_tokens"](input_ids.astype(np.int32))
    )
    inputs_embeds = scatter_vision_into_embeds(input_ids, text_embeds, vision_embeds)
    position_ids = build_multimodal_position_ids(
        input_ids, grid_thw, spatial_merge_size=ms
    )
    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "image_grid_thw": grid_thw,
    }


def qwen2_vl_generate(
    bundle,
    tokenizer,
    image_processor,
    prompt_text: str,
    images=None,
    max_new_tokens: int = 64,
):
    """Greedy generation, returns ``(generated_ids, decoded_text)``."""
    tc = bundle["config"]["text_config"]
    inv_freq = ops.convert_to_tensor(bundle["llm_inv_freq"], dtype="float32")
    eos_id = tokenizer.eos_token_id

    # Encode prompt + images once.
    enc = qwen2_vl_encode_inputs(
        bundle, prompt_text, tokenizer, image_processor, images
    )
    inputs_embeds = enc["inputs_embeds"]
    position_ids = enc["position_ids"]

    generated = []
    for step in range(max_new_tokens):
        T = inputs_embeds.shape[1]
        cos, sin = build_mrope_cos_sin(position_ids, inv_freq, tc["mrope_section"])
        cmask = make_full_causal_mask(T)
        hidden = bundle["llm"](
            {
                "inputs_embeds": inputs_embeds,
                "rope_cos": ops.convert_to_numpy(cos),
                "rope_sin": ops.convert_to_numpy(sin),
                "causal_mask": cmask,
            }
        )
        hidden_np = ops.convert_to_numpy(hidden)
        embed_w = ops.convert_to_numpy(bundle["embed_tokens"].embeddings)
        logits = hidden_np[0, -1] @ embed_w.T
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if next_id == eos_id:
            break

        # Append the new token's embedding and extend position_ids.
        next_embed = embed_w[next_id][None, None, :]
        inputs_embeds = np.concatenate([inputs_embeds, next_embed], axis=1)
        next_pos = position_ids[:, :, -1:] + 1
        position_ids = np.concatenate([position_ids, next_pos], axis=2)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text
