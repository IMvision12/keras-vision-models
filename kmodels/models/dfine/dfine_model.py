import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import get_all_weight_names, load_weights_from_config

from .config import DFINE_MODEL_CONFIG, DFINE_WEIGHTS_CONFIG
from .dfine_layers import (
    DFineDecoderLayer,
    DFineDecoderParams,
    DFineLearnableAffineBlock,
    DFineMultiHeadAttention,
)


def dfine_sine_pos_embed(height, width, embed_dim, temperature=10000):
    """Compute 2D sinusoidal position embedding.

    Generates non-learnable sine/cosine positional encodings for a 2D
    spatial grid. The embedding dimension is split into four equal parts
    encoding height-sin, height-cos, width-sin, and width-cos.

    Reference:
        - `D-FINE <https://arxiv.org/abs/2410.13842>`_

    Args:
        height: Integer, spatial height of the feature map.
        width: Integer, spatial width of the feature map.
        embed_dim: Integer, total embedding dimension. Must be
            divisible by 4.
        temperature: Integer, temperature scaling factor for the
            sinusoidal frequencies. Defaults to ``10000``.

    Returns:
        Tensor of shape ``(1, height * width, embed_dim)``.
    """
    pos_dim = embed_dim // 4
    dim_t = ops.cast(ops.arange(pos_dim), "float32") / pos_dim
    dim_t = 1.0 / (temperature**dim_t)
    grid_w = ops.cast(ops.arange(width), "float32")
    grid_h = ops.cast(ops.arange(height), "float32")
    grid_h, grid_w = ops.meshgrid(grid_h, grid_w, indexing="ij")
    out_w = ops.reshape(grid_w, [-1, 1]) * ops.reshape(dim_t, [1, -1])
    out_h = ops.reshape(grid_h, [-1, 1]) * ops.reshape(dim_t, [1, -1])
    pos = ops.concatenate(
        [ops.sin(out_h), ops.cos(out_h), ops.sin(out_w), ops.cos(out_w)],
        axis=-1,
    )
    return ops.expand_dims(pos, axis=0)


def dfine_conv_bn(
    x,
    out_ch,
    ks,
    stride,
    groups=1,
    padding=None,
    activation="relu",
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Conv + BatchNorm + optional activation + optional LAB block.

    Mirrors HGNetV2ConvLayer from the HuggingFace implementation.

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size.
        stride: Integer, convolution stride.
        groups: Integer, number of convolution groups. Defaults to ``1``.
        padding: Integer or ``None``. If ``None``, defaults to
            ``(ks - 1) // 2``.
        activation: String or ``None``, activation name.
            Defaults to ``"relu"``.
        use_lab: Boolean, whether to apply a Learnable Affine Block
            after activation. Defaults to ``False``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with the same spatial layout convention as
        the input.
    """
    pad = (ks - 1) // 2 if padding is None else padding
    if pad > 0:
        x = layers.ZeroPadding2D(padding=pad, data_format=data_format)(x)
    x = layers.Conv2D(
        out_ch,
        ks,
        strides=stride,
        padding="valid",
        use_bias=False,
        groups=groups,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_bn",
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act")(x)
        if use_lab:
            x = _dfine_lab_layer(x, name=f"{name}_lab")
    return x


def _dfine_lab_layer(x, name=""):
    """Learnable Affine Block: scale * x + bias with scalar parameters.

    Args:
        x: Input tensor.
        name: String, layer name prefix.

    Returns:
        Output tensor with learnable affine transformation applied.
    """
    lab = DFineLearnableAffineBlock(name=name)
    return lab(x)


def dfine_light_conv_block(
    x,
    out_ch,
    ks,
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Light convolution block: 1x1 conv + depthwise conv.

    Mirrors HGNetV2ConvLayerLight.

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size for the depthwise convolution.
        use_lab: Boolean, whether to apply a Learnable Affine Block.
            Defaults to ``False``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor.
    """
    x = dfine_conv_bn(
        x,
        out_ch,
        1,
        1,
        activation=None,
        use_lab=False,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    x = dfine_conv_bn(
        x,
        out_ch,
        ks,
        1,
        groups=out_ch,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    return x


def dfine_basic_layer(
    x,
    mid_ch,
    out_ch,
    num_layers,
    kernel_size=3,
    residual=False,
    light_block=False,
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """HGNetV2 basic layer with sequential convolutions and aggregation.

    All intermediate outputs are concatenated and aggregated through
    two 1x1 convolutions (squeeze + excitation).

    Args:
        x: Input tensor.
        mid_ch: Integer, intermediate channel dimension.
        out_ch: Integer, output channel dimension.
        num_layers: Integer, number of sequential convolutions.
        kernel_size: Integer, convolution kernel size. Defaults to ``3``.
        residual: Boolean, whether to add a residual connection.
            Defaults to ``False``.
        light_block: Boolean, whether to use light convolution blocks.
            Defaults to ``False``.
        use_lab: Boolean, whether to apply Learnable Affine Blocks.
            Defaults to ``False``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor.
    """
    identity = x
    outputs = [x]
    for i in range(num_layers):
        if light_block:
            x = dfine_light_conv_block(
                x,
                mid_ch,
                kernel_size,
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"{name}_layers_{i}",
            )
        else:
            x = dfine_conv_bn(
                x,
                mid_ch,
                kernel_size,
                1,
                activation="relu",
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"{name}_layers_{i}",
            )
        outputs.append(x)
    x = layers.Concatenate(axis=channels_axis, name=f"{name}_cat")(outputs)
    x = dfine_conv_bn(
        x,
        out_ch // 2,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_agg_0",
    )
    x = dfine_conv_bn(
        x,
        out_ch,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_agg_1",
    )
    if residual:
        x = layers.Add(name=f"{name}_add")([x, identity])
    return x


def dfine_backbone(
    input_tensor,
    stem_channels,
    stage_in_channels,
    stage_mid_channels,
    stage_out_channels,
    stage_num_blocks,
    stage_downsample,
    stage_light_block,
    stage_kernel_size,
    stage_numb_of_layers,
    use_lab=False,
    out_stage_indices=None,
    data_format="channels_last",
    channels_axis=-1,
):
    """Build the HGNetV2 backbone for D-FINE.

    Args:
        input_tensor: Keras input tensor (B, H, W, 3).
        stem_channels: List [in_ch, stem_ch, stem_out_ch].
        stage_in_channels: Input channels per backbone stage.
        stage_mid_channels: Middle channels per backbone stage.
        stage_out_channels: Output channels per backbone stage.
        stage_num_blocks: Number of basic blocks per stage.
        stage_downsample: Whether to downsample per stage.
        stage_light_block: Whether to use light blocks per stage.
        stage_kernel_size: Kernel size per stage.
        stage_numb_of_layers: Conv layers per basic block per stage.
        use_lab: Whether to use learnable affine blocks.
        out_stage_indices: List of stage indices to return features from
            (e.g., [2, 3] for nano, [1, 2, 3] for others).
        data_format: String, Keras data format. Defaults to
            ``"channels_last"``.
        channels_axis: Integer, channel axis index. Defaults to ``-1``.

    Returns:
        List of feature tensors from the requested stages.
    """
    if out_stage_indices is None:
        out_stage_indices = [1, 2, 3]

    stem_ch = stem_channels[1]
    stem_out = stem_channels[2]

    x = dfine_conv_bn(
        input_tensor,
        stem_ch,
        3,
        2,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem1",
    )
    x_pad = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(x)
    stem2a = layers.Conv2D(
        stem_ch // 2,
        2,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_stem2a_conv",
    )(x_pad)
    stem2a = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_stem2a_bn",
    )(stem2a)
    stem2a = layers.Activation("relu", name="backbone_stem2a_act")(stem2a)
    if use_lab:
        stem2a = _dfine_lab_layer(stem2a, name="backbone_stem2a_lab")

    stem2a_pad = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(stem2a)
    stem2b = layers.Conv2D(
        stem_ch,
        2,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_stem2b_conv",
    )(stem2a_pad)
    stem2b = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_stem2b_bn",
    )(stem2b)
    stem2b = layers.Activation("relu", name="backbone_stem2b_act")(stem2b)
    if use_lab:
        stem2b = _dfine_lab_layer(stem2b, name="backbone_stem2b_lab")

    pooled = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(x)
    pooled = layers.MaxPooling2D(
        pool_size=2,
        strides=1,
        padding="valid",
        data_format=data_format,
    )(pooled)

    x = layers.Concatenate(
        axis=channels_axis,
        name="backbone_stem_cat",
    )([pooled, stem2b])
    x = dfine_conv_bn(
        x,
        stem_ch,
        3,
        2,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem3",
    )
    x = dfine_conv_bn(
        x,
        stem_out,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem4",
    )

    stage_outputs = []
    for si in range(len(stage_num_blocks)):
        if stage_downsample[si]:
            in_ch_ds = stage_in_channels[si]
            x = dfine_conv_bn(
                x,
                in_ch_ds,
                3,
                2,
                groups=in_ch_ds,
                activation=None,
                use_lab=False,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"backbone_stage{si}_downsample",
            )

        nb = stage_num_blocks[si]
        for bi in range(nb):
            x = dfine_basic_layer(
                x,
                mid_ch=stage_mid_channels[si],
                out_ch=stage_out_channels[si],
                num_layers=stage_numb_of_layers[si],
                kernel_size=stage_kernel_size[si],
                residual=(bi != 0),
                light_block=stage_light_block[si],
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"backbone_stage{si}_block{bi}",
            )
        stage_outputs.append(x)

    return [stage_outputs[i] for i in out_stage_indices]


def dfine_conv_norm(
    x,
    out_ch,
    ks,
    stride,
    groups=1,
    padding=None,
    activation=None,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Convolution + BatchNorm + optional activation block.

    Applies zero-padding, a ``Conv2D`` without bias, batch
    normalization, and an optional activation function. Used as the
    building block for CCFM lateral convolutions, downsample
    convolutions, and RepVGG/CSP blocks.

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size.
        stride: Integer, convolution stride.
        groups: Integer, number of convolution groups. Defaults to ``1``.
        padding: Integer or ``None``. If ``None``, defaults to
            ``(ks - 1) // 2``.
        activation: String or ``None``, activation name.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with the same spatial layout convention as
        the input.
    """
    pad = (ks - 1) // 2 if padding is None else padding
    if pad > 0:
        x = layers.ZeroPadding2D(padding=pad, data_format=data_format)(x)
    x = layers.Conv2D(
        out_ch,
        ks,
        strides=stride,
        padding="valid",
        use_bias=False,
        groups=groups,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_norm",
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act")(x)
    return x


def dfine_rep_vgg_block(
    x,
    ch,
    activation="silu",
    data_format=None,
    channels_axis=-1,
    name="",
):
    """RepVGG block: parallel 3x3 and 1x1 conv branches, summed.

    Args:
        x: Input tensor.
        ch: Integer, number of channels for both branches.
        activation: String, activation name. Defaults to ``"silu"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor of the same shape as the input.
    """
    b1 = dfine_conv_norm(
        x,
        ch,
        3,
        1,
        padding=1,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    b2 = dfine_conv_norm(
        x,
        ch,
        1,
        1,
        padding=0,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    y = layers.Add(name=f"{name}_add")([b1, b2])
    return layers.Activation(activation, name=f"{name}_act")(y)


def dfine_csp_rep_layer(
    x,
    out_ch,
    expansion=1.0,
    num_blocks=1,
    activation="silu",
    data_format=None,
    channels_axis=-1,
    name="",
):
    """CSP Rep Layer: conv1 path through RepVGG bottlenecks + conv2
    shortcut, summed, optionally projected via conv3.

    Args:
        x: Input tensor.
        out_ch: Integer, output channel dimension.
        expansion: Float, hidden channel expansion ratio relative to
            ``out_ch``. Defaults to ``1.0``.
        num_blocks: Integer, number of RepVGG bottleneck blocks.
            Defaults to ``1``.
        activation: String, activation name. Defaults to ``"silu"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``out_ch`` channels.
    """
    hid = int(out_ch * expansion)
    p1 = dfine_conv_norm(
        x,
        hid,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    for i in range(num_blocks):
        p1 = dfine_rep_vgg_block(
            p1,
            hid,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"{name}_bottlenecks_{i}",
        )
    p2 = dfine_conv_norm(
        x,
        hid,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    merged = layers.Add(name=f"{name}_merge")([p1, p2])
    if hid != out_ch:
        merged = dfine_conv_norm(
            merged,
            out_ch,
            1,
            1,
            padding=0,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"{name}_conv3",
        )
    return merged


def dfine_rep_ncspelan4(
    x,
    encoder_hidden_dim,
    hidden_expansion,
    activation="silu",
    num_blocks=1,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """RepNCSPELAN4 block used in D-FINE's FPN and PAN.

    Splits the conv1 output into two halves, processes through two
    CSPRepLayers with intermediate convolutions, then concatenates
    all branches and projects via conv4.

    Args:
        x: Input tensor.
        encoder_hidden_dim: Integer, encoder hidden dimension.
        hidden_expansion: Float, hidden channel expansion ratio.
        activation: String, activation name. Defaults to ``"silu"``.
        num_blocks: Integer, number of RepVGG bottleneck blocks.
            Defaults to ``1``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``encoder_hidden_dim`` channels.
    """
    conv3_dim = encoder_hidden_dim * 2
    conv4_dim = round(hidden_expansion * encoder_hidden_dim // 2)
    conv_dim = conv3_dim // 2

    y = dfine_conv_norm(
        x,
        conv3_dim,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    split_a = (
        y[..., :conv_dim] if data_format == "channels_last" else y[:, :conv_dim, :, :]
    )
    split_b = (
        y[..., conv_dim:] if data_format == "channels_last" else y[:, conv_dim:, :, :]
    )

    branch1 = dfine_csp_rep_layer(
        split_b,
        conv4_dim,
        num_blocks=num_blocks,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_csp_rep1",
    )
    branch1 = dfine_conv_norm(
        branch1,
        conv4_dim,
        3,
        1,
        padding=1,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )

    branch2 = dfine_csp_rep_layer(
        branch1,
        conv4_dim,
        num_blocks=num_blocks,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_csp_rep2",
    )
    branch2 = dfine_conv_norm(
        branch2,
        conv4_dim,
        3,
        1,
        padding=1,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv3",
    )

    merged = layers.Concatenate(axis=channels_axis, name=f"{name}_cat")(
        [split_a, split_b, branch1, branch2]
    )
    out = dfine_conv_norm(
        merged,
        encoder_hidden_dim,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv4",
    )
    return out


def dfine_sc_down(
    x,
    encoder_hidden_dim,
    ks,
    stride,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """SCDown: 1x1 conv + depthwise conv for downsampling in PAN.

    Args:
        x: Input tensor.
        encoder_hidden_dim: Integer, number of output channels.
        ks: Integer, kernel size for the depthwise convolution.
        stride: Integer, stride for the depthwise convolution.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``encoder_hidden_dim`` channels.
    """
    x = dfine_conv_norm(
        x,
        encoder_hidden_dim,
        1,
        1,
        padding=0,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    x = dfine_conv_norm(
        x,
        encoder_hidden_dim,
        ks,
        stride,
        groups=encoder_hidden_dim,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    return x


def dfine_aifi_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    ffn_dim,
    activation="gelu",
    name="aifi_0_layers_0",
):
    """Single AIFI transformer encoder layer for D-FINE.

    Applies self-attention followed by a feedforward network, each with
    a residual connection and post-norm layer normalization. Positional
    embeddings are added to the query and key inputs of self-attention
    but not to the values.

    Args:
        x: Input tensor of shape
            ``(batch_size, seq_len, hidden_dim)``.
        pos_embed: Positional embedding tensor of shape
            ``(1, seq_len, hidden_dim)``, added to the query and key
            inputs of self-attention.
        hidden_dim: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        ffn_dim: Integer, intermediate dimension of the feedforward
            network.
        activation: String, FFN activation function name.
            Defaults to ``"gelu"``.
        name: String, name prefix for all sub-layers in this block.
            Defaults to ``"aifi_0_layers_0"``.

    Returns:
        Output tensor of shape ``(batch_size, seq_len, hidden_dim)``.
    """
    sa = DFineMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_prefix=f"{name}_self_attn",
        name=f"{name}_self_attn",
    )
    q = k = layers.Add(name=f"{name}_sa_qk_add")([x, pos_embed])
    residual = x
    attn = sa(q, k, x)
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name}_self_attn_layer_norm",
    )(layers.Add(name=f"{name}_sa_res")([residual, attn]))
    residual = x
    ff = layers.Dense(ffn_dim, name=f"{name}_fc1")(x)
    ff = layers.Activation(activation, name=f"{name}_gelu")(ff)
    ff = layers.Dense(hidden_dim, name=f"{name}_fc2")(ff)
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name}_final_layer_norm",
    )(layers.Add(name=f"{name}_ff_res")([residual, ff]))
    return x


def dfine_weighting_function(max_num_bins, up, reg_scale):
    """Compute the non-uniform weighting function for FDR.

    Args:
        max_num_bins: Integer, number of distribution bins.
        up: Scalar tensor, learnable upsampling factor.
        reg_scale: Scalar tensor, learnable regression scale.

    Returns:
        Tensor of shape ``(max_num_bins + 1,)`` containing the
        non-uniform weight vector.
    """
    abs_up = ops.abs(up)
    abs_rs = ops.abs(reg_scale)
    upper_bound1 = ops.multiply(abs_up, abs_rs)
    upper_bound2 = ops.multiply(upper_bound1, 2.0)
    step = ops.power(upper_bound1 + 1.0, 2.0 / (max_num_bins - 2))

    values = []
    values.append(ops.reshape(ops.negative(upper_bound2), [1]))
    for i in range(max_num_bins // 2 - 1, 0, -1):
        val = ops.negative(ops.power(step, float(i))) + 1.0
        values.append(ops.reshape(val, [1]))
    values.append(ops.zeros([1], dtype=up.dtype))
    for i in range(1, max_num_bins // 2):
        val = ops.power(step, float(i)) - 1.0
        values.append(ops.reshape(val, [1]))
    values.append(ops.reshape(upper_bound2, [1]))

    return ops.concatenate(values, axis=0)


def dfine_integral(pred_corners, project, max_num_bins):
    """Apply DFine integral to convert distribution to distances.

    Args:
        pred_corners: Tensor of shape ``(B, Q, 4 * (max_num_bins + 1))``.
        project: Tensor of shape ``(max_num_bins + 1,)``.
        max_num_bins: Integer, number of bins.

    Returns:
        Tensor of shape ``(B, Q, 4)`` with distances.
    """
    nbins = max_num_bins + 1
    orig_shape = pred_corners.shape
    flat = ops.reshape(pred_corners, [-1, nbins])
    flat = ops.softmax(flat, axis=1)
    proj = ops.reshape(project, [nbins, 1])
    flat = ops.matmul(flat, proj)
    flat = ops.reshape(flat, [-1, 4])
    return ops.reshape(flat, [-1, orig_shape[1], 4])


def dfine_distance2bbox(points, distance, reg_scale):
    """Convert reference points + distances to cxcywh bounding boxes.

    Args:
        points: Tensor ``(B, Q, 4)`` as ``(cx, cy, w, h)`` in [0, 1].
        distance: Tensor ``(B, Q, 4)`` with distances.
        reg_scale: Scalar tensor, regression scale.

    Returns:
        Tensor ``(B, Q, 4)`` in ``(cx, cy, w, h)`` format.
    """
    rs = ops.abs(reg_scale)
    half_rs = ops.multiply(ops.convert_to_tensor(0.5, dtype=rs.dtype), rs)
    pw = ops.divide(points[..., 2], rs)
    ph = ops.divide(points[..., 3], rs)
    x1 = ops.subtract(
        points[..., 0], ops.multiply(ops.add(half_rs, distance[..., 0]), pw)
    )
    y1 = ops.subtract(
        points[..., 1], ops.multiply(ops.add(half_rs, distance[..., 1]), ph)
    )
    x2 = ops.add(points[..., 0], ops.multiply(ops.add(half_rs, distance[..., 2]), pw))
    y2 = ops.add(points[..., 1], ops.multiply(ops.add(half_rs, distance[..., 3]), ph))
    cx = ops.divide(ops.add(x1, x2), 2.0)
    cy = ops.divide(ops.add(y1, y2), 2.0)
    w = ops.subtract(x2, x1)
    h = ops.subtract(y2, y1)
    return ops.stack([cx, cy, w, h], axis=-1)


def dfine_inverse_sigmoid(t, eps=1e-5):
    """Numerically stable inverse sigmoid.

    Args:
        t: Input tensor with values in (0, 1).
        eps: Float, clamping epsilon. Defaults to ``1e-5``.

    Returns:
        Tensor with inverse sigmoid applied.
    """
    t = ops.clip(t, eps, 1 - eps)
    return ops.log(t / (1 - t))


@keras.saving.register_keras_serializable(package="kmodels")
class DFine(keras.Model):
    """D-FINE: Detection with Fine-grained Distribution Refinement.

    A real-time object detection model combining an HGNetV2 backbone with
    a hybrid encoder (AIFI + CCFM) and a decoder with Fine-grained
    Distribution Refinement (FDR) and Localization Quality Estimation
    (LQE).

    Reference:
        - `D-FINE: Redefine Regression Task of DETRs as Fine-grained
          Distribution Refinement <https://arxiv.org/abs/2410.13842>`_

    Args:
        stem_channels: Stem channel configuration ``[in, mid, out]``.
        stage_in_channels: Input channels per backbone stage.
        stage_mid_channels: Middle channels per backbone stage.
        stage_out_channels: Output channels per backbone stage.
        stage_num_blocks: Number of basic blocks per stage.
        stage_downsample: Whether to downsample per stage.
        stage_light_block: Whether to use light blocks per stage.
        stage_kernel_size: Kernel size per stage.
        stage_numb_of_layers: Conv layers per basic block per stage.
        use_lab: Whether to use Learnable Affine Block.
        encoder_in_channels: Backbone channels fed to encoder.
        encoder_hidden_dim: Hidden dim of hybrid encoder.
        encoder_layers: Number of AIFI encoder layers.
        encoder_ffn_dim: FFN dim in AIFI encoder.
        encoder_num_heads: Attention heads in AIFI encoder.
        encode_proj_layers: Feature level indices for AIFI.
        encoder_activation_function: Activation in the AIFI FFN.
        activation_function: Activation in CCFM (FPN/PAN) blocks.
        hidden_expansion: CSP hidden channel expansion ratio.
        d_model: Decoder model dimension.
        decoder_layers: Number of decoder layers.
        decoder_ffn_dim: FFN dim in decoder.
        decoder_num_heads: Attention heads in decoder.
        decoder_n_points: List of sampling points per feature level.
        decoder_activation_function: Activation in the decoder FFN.
        num_feature_levels: Number of multi-scale feature levels.
        feat_strides: Feature strides from backbone.
        max_num_bins: Number of FDR distribution bins.
        num_queries: Number of object queries.
        num_labels: Number of object classes.
        weights: Pre-trained weight identifier or file path.
        input_shape: Input image shape ``(H, W, C)``.
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    STAGE_DOWNSAMPLE = (False, True, True, True)
    STAGE_LIGHT_BLOCK = (False, False, True, True)
    STAGE_KERNEL_SIZE = (3, 3, 5, 5)
    ENCODER_LAYERS = 1
    ENCODER_NUM_HEADS = 8
    ENCODER_ACTIVATION = "gelu"
    ACTIVATION = "silu"
    DECODER_NUM_HEADS = 8
    DECODER_ACTIVATION = "relu"
    MAX_NUM_BINS = 32

    def __init__(
        self,
        stem_channels=(3, 16, 16),
        stage_in_channels=(16, 64, 256, 512),
        stage_mid_channels=(16, 32, 64, 128),
        stage_out_channels=(64, 256, 512, 1024),
        stage_num_blocks=(1, 1, 2, 1),
        stage_numb_of_layers=(3, 3, 3, 3),
        use_lab=True,
        encoder_in_channels=(256, 512, 1024),
        encoder_hidden_dim=256,
        encoder_ffn_dim=1024,
        encode_proj_layers=(2,),
        hidden_expansion=1.0,
        d_model=256,
        decoder_layers=6,
        decoder_ffn_dim=1024,
        decoder_n_points=None,
        num_feature_levels=3,
        feat_strides=(8, 16, 32),
        num_queries=300,
        num_labels=80,
        weights="coco",
        input_shape=None,
        input_tensor=None,
        name="DFine",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        if input_shape is None:
            input_shape = (640, 640, 3)
        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(
                    tensor=input_tensor,
                    shape=input_shape,
                )
            else:
                img_input = input_tensor
        inputs = img_input

        if decoder_n_points is None:
            decoder_n_points = [4] * num_feature_levels

        out_stage_indices = []
        for enc_ch in encoder_in_channels:
            for si, soc in enumerate(stage_out_channels):
                if soc == enc_ch and si not in out_stage_indices:
                    out_stage_indices.append(si)
                    break

        bk_feats = dfine_backbone(
            inputs,
            stem_channels=list(stem_channels),
            stage_in_channels=list(stage_in_channels),
            stage_mid_channels=list(stage_mid_channels),
            stage_out_channels=list(stage_out_channels),
            stage_num_blocks=list(stage_num_blocks),
            stage_downsample=list(self.STAGE_DOWNSAMPLE),
            stage_light_block=list(self.STAGE_LIGHT_BLOCK),
            stage_kernel_size=list(self.STAGE_KERNEL_SIZE),
            stage_numb_of_layers=list(stage_numb_of_layers),
            use_lab=use_lab,
            out_stage_indices=out_stage_indices,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        proj_feats = []
        for i, feat in enumerate(bk_feats):
            p = layers.Conv2D(
                encoder_hidden_dim,
                1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"encoder_input_proj_{i}_conv",
            )(feat)
            p = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"encoder_input_proj_{i}_bn",
            )(p)
            proj_feats.append(p)

        if data_format == "channels_first":
            spatial_h, spatial_w = input_shape[1], input_shape[2]
        else:
            spatial_h, spatial_w = input_shape[0], input_shape[1]

        for ai, enc_lvl in enumerate(encode_proj_layers):
            feat = proj_feats[enc_lvl]
            if data_format == "channels_first":
                feat = layers.Permute(
                    (2, 3, 1),
                    name=f"aifi_{ai}_to_cl",
                )(feat)
            h = spatial_h // feat_strides[enc_lvl]
            w = spatial_w // feat_strides[enc_lvl]
            flat = layers.Reshape(
                (h * w, encoder_hidden_dim),
                name=f"aifi_{ai}_flatten",
            )(feat)
            pe = dfine_sine_pos_embed(h, w, encoder_hidden_dim, 10000)
            for li in range(self.ENCODER_LAYERS):
                flat = dfine_aifi_encoder_layer(
                    flat,
                    pe,
                    encoder_hidden_dim,
                    self.ENCODER_NUM_HEADS,
                    encoder_ffn_dim,
                    activation=self.ENCODER_ACTIVATION,
                    name=f"aifi_{ai}_layers_{li}",
                )
            unflat = layers.Reshape(
                (h, w, encoder_hidden_dim),
                name=f"aifi_{ai}_unflatten",
            )(flat)
            if data_format == "channels_first":
                unflat = layers.Permute(
                    (3, 1, 2),
                    name=f"aifi_{ai}_to_cf",
                )(unflat)
            proj_feats[enc_lvl] = unflat

        num_fpn = num_feature_levels - 1
        fpn = [proj_feats[-1]]
        for idx in range(num_fpn):
            bk_feat = proj_feats[num_fpn - idx - 1]
            top = fpn[-1]
            top = dfine_conv_norm(
                top,
                encoder_hidden_dim,
                1,
                1,
                padding=0,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"lateral_convs_{idx}",
            )
            fpn[-1] = top
            top = layers.UpSampling2D(
                size=2,
                interpolation="nearest",
                data_format=data_format,
                name=f"fpn_up_{idx}",
            )(top)
            fused = layers.Concatenate(
                axis=channels_axis,
                name=f"fpn_cat_{idx}",
            )([top, bk_feat])
            fpn.append(
                dfine_rep_ncspelan4(
                    fused,
                    encoder_hidden_dim,
                    hidden_expansion,
                    activation=self.ACTIVATION,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    name=f"fpn_blocks_{idx}",
                )
            )
        fpn.reverse()

        pan = [fpn[0]]
        for idx in range(num_fpn):
            top_pan = pan[-1]
            fpn_feat = fpn[idx + 1]
            down = dfine_sc_down(
                top_pan,
                encoder_hidden_dim,
                3,
                2,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"downsample_convs_{idx}",
            )
            fused = layers.Concatenate(
                axis=channels_axis,
                name=f"pan_cat_{idx}",
            )([down, fpn_feat])
            pan.append(
                dfine_rep_ncspelan4(
                    fused,
                    encoder_hidden_dim,
                    hidden_expansion,
                    activation=self.ACTIVATION,
                    data_format=data_format,
                    channels_axis=channels_axis,
                    name=f"pan_blocks_{idx}",
                )
            )

        dec_sources = []
        for i, feat in enumerate(pan):
            if d_model != encoder_hidden_dim:
                p = layers.Conv2D(
                    d_model,
                    1,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"decoder_input_proj_{i}_conv",
                )(feat)
                p = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"decoder_input_proj_{i}_bn",
                )(p)
                dec_sources.append(p)
            else:
                dec_sources.append(feat)

        spatial_shapes = [(spatial_h // s, spatial_w // s) for s in feat_strides]
        flat_list = []
        for i, src in enumerate(dec_sources):
            hi, wi = spatial_shapes[i]
            if data_format == "channels_first":
                src = layers.Permute(
                    (2, 3, 1),
                    name=f"dec_flat_{i}_to_cl",
                )(src)
            flat_list.append(
                layers.Reshape(
                    (hi * wi, d_model),
                    name=f"dec_flat_{i}",
                )(src)
            )
        source_flat = layers.Concatenate(
            axis=1,
            name="dec_src_cat",
        )(flat_list)

        level_start = []
        cum = 0
        for hi, wi in spatial_shapes:
            level_start.append(cum)
            cum += hi * wi

        gs = 0.05
        anc_parts = []
        for lvl, (hi, wi) in enumerate(spatial_shapes):
            gy, gx = ops.meshgrid(
                ops.cast(ops.arange(hi), "float32"),
                ops.cast(ops.arange(wi), "float32"),
                indexing="ij",
            )
            xy = ops.reshape(
                ops.stack([gx, gy], axis=-1),
                [1, hi * wi, 2],
            )
            xy = (xy + 0.5) / ops.convert_to_tensor(
                [[[float(wi), float(hi)]]],
                dtype="float32",
            )
            wh = ops.ones_like(xy) * gs * (2.0**lvl)
            anc_parts.append(ops.concatenate([xy, wh], axis=-1))
        anchors = ops.concatenate(anc_parts, axis=1)
        vmask = ops.cast(
            ops.all(
                (anchors > 1e-2) & (anchors < 1 - 1e-2),
                axis=-1,
                keepdims=True,
            ),
            "float32",
        )
        anc_logit = ops.where(
            vmask > 0.5,
            ops.log(anchors / (1 - anchors)),
            ops.convert_to_tensor(3.4028235e38, dtype="float32"),
        )
        anchors_t = anc_logit
        vmask_t = vmask

        memory = source_flat * vmask_t
        enc_out = layers.Dense(d_model, name="enc_output_linear")(memory)
        enc_out = layers.LayerNormalization(
            epsilon=1e-5,
            name="enc_output_layernorm",
        )(enc_out)
        enc_scores = layers.Dense(num_labels, name="enc_score_head")(enc_out)
        enc_bb = layers.Dense(
            d_model,
            activation="relu",
            name="enc_bbox_head_0",
        )(enc_out)
        enc_bb = layers.Dense(
            d_model,
            activation="relu",
            name="enc_bbox_head_1",
        )(enc_bb)
        enc_bb = layers.Dense(4, name="enc_bbox_head_2")(enc_bb)
        enc_bb_logits = enc_bb + anchors_t

        max_sc = ops.max(enc_scores, axis=-1)
        _, topk_idx = ops.top_k(max_sc, k=num_queries)
        idx3 = ops.expand_dims(topk_idx, -1)
        target = ops.take_along_axis(enc_out, idx3, axis=1)
        target = ops.stop_gradient(target)
        idx4 = ops.repeat(idx3, 4, axis=-1)
        ref_logit = ops.take_along_axis(enc_bb_logits, idx4, axis=1)
        ref_logit = ops.stop_gradient(ref_logit)

        decoder_params = DFineDecoderParams(name="decoder_params")

        qp_d0 = layers.Dense(
            d_model * 2,
            activation="relu",
            name="query_pos_head_0",
        )
        qp_d1 = layers.Dense(d_model, name="query_pos_head_1")

        pre_bb_d0 = layers.Dense(
            d_model,
            activation="relu",
            name="pre_bbox_head_0",
        )
        pre_bb_d1 = layers.Dense(
            d_model,
            activation="relu",
            name="pre_bbox_head_1",
        )
        pre_bb_d2 = layers.Dense(4, name="pre_bbox_head_2")

        hs = target
        ref_pts = ops.sigmoid(ref_logit)
        ref_detach = ops.stop_gradient(ref_pts)

        output_detach = ops.zeros_like(hs)
        max_num_bins = self.MAX_NUM_BINS
        nbins_out = 4 * (max_num_bins + 1)
        pred_corners_accum = None

        ref_points_initial = None
        all_logits = []
        last_boxes = None

        for di in range(decoder_layers):
            rp_in = ops.expand_dims(ref_detach, axis=2)
            query_pos = qp_d1(qp_d0(ref_detach))
            query_pos = ops.clip(query_pos, -10.0, 10.0)

            dl = DFineDecoderLayer(
                d_model=d_model,
                num_heads=self.DECODER_NUM_HEADS,
                dim_feedforward=decoder_ffn_dim,
                activation=self.DECODER_ACTIVATION,
                n_levels=num_feature_levels,
                num_points_list=decoder_n_points,
                offset_scale=0.5,
                spatial_shapes=spatial_shapes,
                block_prefix=f"decoder_layers_{di}",
                name=f"decoder_layers_{di}",
            )
            hs = dl(hs, source_flat, query_pos, rp_in)

            if di == 0:
                hs = decoder_params(hs)
                pre_bb = pre_bb_d2(pre_bb_d1(pre_bb_d0(hs)))
                new_ref = ops.sigmoid(pre_bb + dfine_inverse_sigmoid(ref_detach))
                ref_points_initial = ops.stop_gradient(new_ref)

            bb_i = layers.Dense(
                d_model,
                activation="relu",
                name=f"bbox_embed_{di}_0",
            )(hs + output_detach)
            bb_i = layers.Dense(
                d_model,
                activation="relu",
                name=f"bbox_embed_{di}_1",
            )(bb_i)
            bb_i = layers.Dense(
                nbins_out,
                name=f"bbox_embed_{di}_2",
            )(bb_i)
            pred_corners = (
                bb_i if pred_corners_accum is None else bb_i + pred_corners_accum
            )

            up_val = decoder_params.up
            rs_val = decoder_params.reg_scale
            project = dfine_weighting_function(max_num_bins, up_val, rs_val)
            distances = dfine_integral(pred_corners, project, max_num_bins)
            inter_ref_bbox = dfine_distance2bbox(
                ref_points_initial,
                distances,
                rs_val,
            )

            pred_corners_accum = pred_corners
            ref_detach = ops.stop_gradient(inter_ref_bbox)
            output_detach = ops.stop_gradient(hs)

            logits_i = layers.Dense(
                num_labels,
                name=f"class_embed_{di}",
            )(hs)

            prob = ops.softmax(
                ops.reshape(pred_corners, [-1, num_queries, 4, max_num_bins + 1]),
                axis=-1,
            )
            prob_topk, _ = ops.top_k(prob, k=4)
            prob_mean = ops.mean(prob_topk, axis=-1, keepdims=True)
            stat = ops.concatenate([prob_topk, prob_mean], axis=-1)
            stat = ops.reshape(stat, [-1, num_queries, 4 * 5])
            quality_score = layers.Dense(
                64,
                activation="relu",
                name=f"lqe_{di}_0",
            )(stat)
            quality_score = layers.Dense(1, name=f"lqe_{di}_1")(quality_score)
            logits_i = logits_i + quality_score

            all_logits.append(logits_i)
            last_boxes = inter_ref_bbox

        last_logits = all_logits[-1]
        for prev_logits in all_logits[:-1]:
            last_logits = last_logits + 0.0 * prev_logits

        outputs = {"logits": last_logits, "pred_boxes": last_boxes}
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self._stem_channels = list(stem_channels)
        self._stage_in_channels = list(stage_in_channels)
        self._stage_mid_channels = list(stage_mid_channels)
        self._stage_out_channels = list(stage_out_channels)
        self._stage_num_blocks = list(stage_num_blocks)
        self._stage_numb_of_layers = list(stage_numb_of_layers)
        self._use_lab = use_lab
        self._encoder_in_channels = list(encoder_in_channels)
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_ffn_dim = encoder_ffn_dim
        self._encode_proj_layers = list(encode_proj_layers)
        self._hidden_expansion = hidden_expansion
        self._d_model = d_model
        self._decoder_layers = decoder_layers
        self._decoder_ffn_dim = decoder_ffn_dim
        self._decoder_n_points = list(decoder_n_points)
        self._num_feature_levels = num_feature_levels
        self._feat_strides = list(feat_strides)
        self._num_queries = num_queries
        self._num_labels = num_labels
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_channels": self._stem_channels,
                "stage_in_channels": self._stage_in_channels,
                "stage_mid_channels": self._stage_mid_channels,
                "stage_out_channels": self._stage_out_channels,
                "stage_num_blocks": self._stage_num_blocks,
                "stage_numb_of_layers": self._stage_numb_of_layers,
                "use_lab": self._use_lab,
                "encoder_in_channels": self._encoder_in_channels,
                "encoder_hidden_dim": self._encoder_hidden_dim,
                "encoder_ffn_dim": self._encoder_ffn_dim,
                "encode_proj_layers": self._encode_proj_layers,
                "hidden_expansion": self._hidden_expansion,
                "d_model": self._d_model,
                "decoder_layers": self._decoder_layers,
                "decoder_ffn_dim": self._decoder_ffn_dim,
                "decoder_n_points": self._decoder_n_points,
                "num_feature_levels": self._num_feature_levels,
                "feat_strides": self._feat_strides,
                "num_queries": self._num_queries,
                "num_labels": self._num_labels,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_dfine_model(
    variant,
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name=None,
    **kwargs,
):
    """Factory function for creating D-FINE model variants.

    Looks up the architecture configuration for the given variant name,
    instantiates a ``DFine`` model, and optionally loads pretrained
    weights from the configured URL or a local file path.

    Args:
        variant: String, model variant name (e.g., ``"DFineSmall"``).
        num_queries: Integer, number of object queries.
            Defaults to ``300``.
        num_labels: Integer, number of object classes (COCO: 80).
            Defaults to ``80``.
        weights: String, one of ``None``, a weight identifier from the
            config (e.g., ``"coco"``), or a path to a weights file.
            Defaults to ``"coco"``.
        input_shape: Optional tuple of integers specifying the input
            shape. Defaults to ``(640, 640, 3)``.
        input_tensor: Optional Keras tensor to use as the model input.
        name: String, the name of the model.
        **kwargs: Additional keyword arguments passed to ``DFine``.

    Returns:
        A configured ``DFine`` model instance.
    """
    cfg = DFINE_MODEL_CONFIG[variant]
    if input_shape is None:
        input_shape = (640, 640, 3)
    model = DFine(
        stem_channels=cfg["stem_channels"],
        stage_in_channels=cfg["stage_in_channels"],
        stage_mid_channels=cfg["stage_mid_channels"],
        stage_out_channels=cfg["stage_out_channels"],
        stage_num_blocks=cfg["stage_num_blocks"],
        stage_numb_of_layers=cfg["stage_numb_of_layers"],
        use_lab=cfg.get("use_lab", True),
        encoder_in_channels=cfg["encoder_in_channels"],
        encoder_hidden_dim=cfg.get("encoder_hidden_dim", 256),
        encoder_ffn_dim=cfg.get("encoder_ffn_dim", 1024),
        encode_proj_layers=cfg.get("encode_proj_layers", (2,)),
        hidden_expansion=cfg.get("hidden_expansion", 1.0),
        d_model=cfg.get("d_model", 256),
        decoder_layers=cfg["decoder_layers"],
        decoder_ffn_dim=cfg.get("decoder_ffn_dim", 1024),
        decoder_n_points=cfg.get("decoder_n_points", [4, 4, 4]),
        num_feature_levels=cfg.get("num_feature_levels", 3),
        feat_strides=cfg.get("feat_strides", (8, 16, 32)),
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name or variant,
        **kwargs,
    )
    if weights in get_all_weight_names(DFINE_WEIGHTS_CONFIG):
        load_weights_from_config(variant, weights, model, DFINE_WEIGHTS_CONFIG)
    elif weights is not None:
        model.load_weights(weights)
    else:
        print("No weights loaded.")
    return model


@register_model
def DFineNano(
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DFineNano",
    **kwargs,
):
    return _create_dfine_model(
        "DFineNano",
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def DFineSmall(
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DFineSmall",
    **kwargs,
):
    return _create_dfine_model(
        "DFineSmall",
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def DFineMedium(
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DFineMedium",
    **kwargs,
):
    return _create_dfine_model(
        "DFineMedium",
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def DFineLarge(
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DFineLarge",
    **kwargs,
):
    return _create_dfine_model(
        "DFineLarge",
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def DFineXLarge(
    num_queries=300,
    num_labels=80,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="DFineXLarge",
    **kwargs,
):
    return _create_dfine_model(
        "DFineXLarge",
        num_queries=num_queries,
        num_labels=num_labels,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )
