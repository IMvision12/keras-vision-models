from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _get_axes(ax, figsize=(8, 8)):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_image(image, ax=None, title: Optional[str] = None, figsize=(8, 8)):
    """Display an image on an axis with the axes hidden.

    Args:
        image: Any array-like that ``matplotlib.pyplot.imshow`` accepts.
        ax: Optional existing ``matplotlib.axes.Axes``. A new figure is
            created when ``None``.
        title: Optional title shown above the axis.
        figsize: Used only when a new figure is created.
    """
    ax = _get_axes(ax, figsize=figsize)
    ax.imshow(np.asarray(image))
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_boxes(
    boxes,
    labels: Optional[Sequence] = None,
    scores: Optional[Sequence[float]] = None,
    ax=None,
    color: str = "red",
    linewidth: float = 2.0,
    fontsize: int = 9,
):
    """Draw ``xyxy`` bounding boxes on an axis.

    Args:
        boxes: ``(N, 4)`` array-like of ``[x0, y0, x1, y1]`` in pixel space.
        labels: Optional per-box label strings.
        scores: Optional per-box confidence scores (rendered to 2 decimals).
        ax: Target axis.
        color: Edge color for the rectangles.
        linewidth: Rectangle edge width.
        fontsize: Font size for label text.
    """
    import matplotlib.patches as patches

    ax = _get_axes(ax)
    boxes = np.asarray(boxes).reshape(-1, 4)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text_parts = []
        if labels is not None:
            text_parts.append(str(labels[i]))
        if scores is not None:
            text_parts.append(f"{float(scores[i]):.2f}")
        if text_parts:
            ax.text(
                x0,
                max(y0 - 4, 0),
                " ".join(text_parts),
                color="white",
                fontsize=fontsize,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1},
            )

    return ax


def plot_masks(
    masks,
    ax=None,
    colors=None,
    alpha: float = 0.5,
):
    """Overlay one or more binary masks on an axis.

    Args:
        masks: ``(H, W)`` single mask or ``(N, H, W)`` stack of binary masks
            (boolean or 0/1 int/float).
        ax: Target axis.
        colors: Optional list of per-mask RGB(A) colors. Defaults to the
            ``tab20`` colormap cycled across masks.
        alpha: Fill transparency for non-background pixels.
    """
    import matplotlib.pyplot as plt

    ax = _get_axes(ax)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks of shape (H,W) or (N,H,W), got {masks.shape}."
        )

    n, h, w = masks.shape
    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(n)]

    for mask, color in zip(masks, colors):
        rgb = np.asarray(color)[:3]
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        active = mask > 0
        overlay[active, :3] = rgb
        overlay[active, 3] = alpha
        ax.imshow(overlay)

    return ax


def plot_points(
    points,
    labels=None,
    ax=None,
    pos_color: str = "lime",
    neg_color: str = "red",
    marker_size: int = 120,
):
    """Overlay SAM-style point prompts.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
        labels: Optional ``(N,)`` array where ``1`` is a positive (foreground)
            prompt and ``0`` is a negative (background) prompt. Defaults to
            all-positive.
        ax: Target axis.
        pos_color: Marker color for positive points.
        neg_color: Marker color for negative points.
        marker_size: Scatter marker size.
    """
    ax = _get_axes(ax)
    points = np.asarray(points).reshape(-1, 2)
    if labels is None:
        labels = np.ones(len(points), dtype=int)
    labels = np.asarray(labels).reshape(-1)

    pos = points[labels == 1]
    neg = points[labels == 0]

    if len(pos):
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=pos_color,
            s=marker_size,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
        )
    if len(neg):
        ax.scatter(
            neg[:, 0],
            neg[:, 1],
            c=neg_color,
            s=marker_size,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
        )
    return ax


def overlay_depth(
    depth,
    cmap: str = "inferno",
    ax=None,
    title: Optional[str] = None,
    figsize=(8, 8),
):
    """Display a depth map with min-max normalization and a colormap.

    Args:
        depth: ``(H, W)`` array of depth values.
        cmap: Matplotlib colormap name.
        ax: Target axis.
        title: Optional title.
        figsize: Used only when a new figure is created.
    """
    ax = _get_axes(ax, figsize=figsize)
    depth = np.asarray(depth, dtype=np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    ax.imshow(depth, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax
