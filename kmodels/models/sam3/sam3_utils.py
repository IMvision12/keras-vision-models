"""SAM3 utility functions: math ops, box conversions, positional encodings."""

import math

import numpy as np
from keras import ops

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def inverse_sigmoid(x, eps=1e-3):
    x = ops.clip(x, eps, 1.0 - eps)
    return ops.log(x / (1.0 - x))


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return ops.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def box_xyxy_to_cxcywh(boxes):
    x0, y0, x1, y1 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return ops.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def scale_boxes(boxes, target_size):
    h, w = target_size
    return boxes * np.array([w, h, w, h], dtype=np.float32)


def compute_scores(pred_logits, presence_logits):
    scores = sigmoid(pred_logits)
    if presence_logits is not None:
        presence = np.asarray(presence_logits)
        if presence.ndim == 1:
            p = sigmoid(presence[-1:]).reshape(1, 1)
        else:
            p = sigmoid(presence)
        scores = scores * p
    return scores


def rotate_pairwise(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = ops.stack([-x2, x1], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(rotated, shape)


def apply_rotary_pos_emb_2d(q, k, cos, sin):
    q_embed = q * cos + rotate_pairwise(q) * sin
    k_embed = k * cos + rotate_pairwise(k) * sin
    return q_embed, k_embed


def sine_encode_boxes(boxes, num_pos_feats=128, temperature=10000):
    """Encode box coords matching HF: interleaved sin/cos, order (y, x, w, h)."""
    scale = 2 * 3.141592653589793
    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    def _encode_coord(coord):
        c = coord * scale
        c = ops.expand_dims(c, axis=-1) / dim_t
        c_sin = ops.sin(c[..., 0::2])
        c_cos = ops.cos(c[..., 1::2])
        half = num_pos_feats // 2
        parts = []
        for j in range(half):
            parts.append(c_sin[:, :, j : j + 1])
            parts.append(c_cos[:, :, j : j + 1])
        return ops.concatenate(parts, axis=-1)

    pos_y = _encode_coord(boxes[:, :, 1])
    pos_x = _encode_coord(boxes[:, :, 0])
    pos_w = _encode_coord(boxes[:, :, 2])
    pos_h = _encode_coord(boxes[:, :, 3])
    return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)


def compute_sine_pos_encoding(
    height,
    width,
    num_pos_feats,
    temperature=10000,
    normalize=True,
    data_format="channels_last",
):
    """2D sine position encoding grid.

    Returns:
        channels_first: (1, 2*num_pos_feats, H, W)
        channels_last:  (1, H, W, 2*num_pos_feats)
    """
    scale = 2 * math.pi
    ones = ops.ones((1, height, width), dtype="float32")
    y_embed = ops.cumsum(ones, axis=1)
    x_embed = ops.cumsum(ones, axis=2)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
    pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

    pos_x_sin = ops.sin(pos_x[:, :, :, 0::2])
    pos_x_cos = ops.cos(pos_x[:, :, :, 1::2])
    pos_y_sin = ops.sin(pos_y[:, :, :, 0::2])
    pos_y_cos = ops.cos(pos_y[:, :, :, 1::2])

    pos_x = ops.reshape(
        ops.stack([pos_x_sin, pos_x_cos], axis=4),
        (1, height, width, num_pos_feats),
    )
    pos_y = ops.reshape(
        ops.stack([pos_y_sin, pos_y_cos], axis=4),
        (1, height, width, num_pos_feats),
    )

    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    if data_format == "channels_first":
        pos = ops.transpose(pos, (0, 3, 1, 2))
    return pos


def resize_mask(mask, target_h, target_w):
    """Resize a single 2D mask to target size using PIL bilinear."""
    if Image is None:
        raise ImportError("PIL required for mask resizing")
    pil_m = Image.fromarray((mask * 255).astype(np.uint8))
    pil_m = pil_m.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_m, dtype=np.float32) / 255.0


def resize_masks_batch(masks, target_h, target_w):
    """Resize a batch of masks (N, H, W) to (N, target_h, target_w)."""
    if len(masks) == 0:
        return masks
    return np.stack([resize_mask(m, target_h, target_w) for m in masks])


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def draw_detections(image, results, title=""):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_instance_masks(image, results, title=""):
    vis = np.array(image, dtype=np.float32)
    for i in range(len(results["scores"])):
        mask = np.asarray(results["masks"][i])
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        vis[mask > 0] = vis[mask > 0] * 0.5 + color * 0.5

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_semantic_mask(image, mask, title="", color=(0, 200, 255)):
    vis = np.array(image, dtype=np.float32)
    mask = np.asarray(mask)
    vis[mask > 0] = vis[mask > 0] * 0.4 + np.array(color, dtype=np.float32) * 0.6

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    pct = 100 * mask.sum() / mask.size
    label = f"{title} ({pct:.1f}%)" if title else f"{pct:.1f}%"
    draw.text((10, 10), label, fill=(255, 255, 255))
    return vis
