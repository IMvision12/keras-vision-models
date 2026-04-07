"""SAM3 utility functions: math ops, box conversions, positional encodings."""

import math

from keras import ops


def inverse_sigmoid(x, eps=1e-3):
    x = ops.clip(x, eps, 1.0 - eps)
    return ops.log(x / (1.0 - x))


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = ops.split(boxes, 4, axis=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return ops.concatenate([x0, y0, x1, y1], axis=-1)


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
    height, width, num_pos_feats, temperature=10000, normalize=True
):
    """2D sine position encoding grid. Returns (1, 2*num_pos_feats, H, W)."""
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
    pos = ops.transpose(pos, (0, 3, 1, 2))
    return pos
