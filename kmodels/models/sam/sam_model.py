import keras
from keras import layers, utils

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import SAM_MODEL_CONFIG, SAM_WEIGHTS_CONFIG
from .sam_layers import (
    SAMAbsolutePositionEmbedding,
    SAMImagePositionalEmbeddings,
    SAMMaskDecoderLayer,
    SAMPositionalEmbedding,
    SAMPromptEncoderLayer,
    SAMVisionLayer,
)


def sam_vision_neck(
    inputs, output_channels, data_format="channels_last", name="vision_encoder_neck"
):
    """Projection neck from vision encoder to mask decoder dimension.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        inputs: Input tensor from the vision encoder.
        output_channels: int, output channel dimension.
        data_format: string, image data format. Defaults to ``"channels_last"``.
        name: string, name prefix for sub-layers.
            Defaults to ``"vision_encoder_neck"``.

    Returns:
        Output tensor.
    """
    x = layers.Conv2D(
        output_channels,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv1",
    )(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm1")(x)
    x = layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm2")(x)
    return x


def sam_feed_forward(
    inputs, hidden_dim, output_dim, num_layers, sigmoid_output=False, name=""
):
    """Multi-layer perceptron for mask decoder prediction heads.

    Builds a feedforward network with ReLU activations between
    hidden layers. Used for the IoU prediction head and the
    output hypernetwork MLPs that generate per-mask dynamic
    convolution weights.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        inputs: Input tensor of shape
            ``(batch_size, ..., input_dim)``.
        hidden_dim: Integer, hidden dimension for all
            intermediate layers.
        output_dim: Integer, output dimension of the final
            linear layer.
        num_layers: Integer, total number of linear layers
            (including input and output projections).
        sigmoid_output: Boolean, whether to apply sigmoid
            activation to the final output.
            Defaults to ``False``.
        name: String, name prefix for all sub-layers.
            Defaults to ``""``.

    Returns:
        Output tensor of shape
        ``(batch_size, ..., output_dim)``.
    """
    x = layers.Dense(hidden_dim, name=f"{name}_proj_in")(inputs)
    x = layers.Activation("relu", name=f"{name}_relu_0")(x)
    for i in range(num_layers - 2):
        x = layers.Dense(hidden_dim, name=f"{name}_layers_{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu_{i + 1}")(x)
    x = layers.Dense(output_dim, name=f"{name}_proj_out")(x)
    if sigmoid_output:
        x = layers.Activation("sigmoid", name=f"{name}_sigmoid")(x)
    return x


def sam_mask_embedding(
    inputs,
    hidden_size=256,
    mask_input_channels=16,
    layer_norm_eps=1e-6,
    data_format="channels_last",
    name="mask_embed",
):
    """Embeds dense mask prompts through a small convolutional network.

    Two stride-2 convolutions (each followed by a channel-axis layer
    norm and GELU) downsample the input mask by 4×, then a 1×1
    convolution projects to the prompt encoder hidden size. Mirrors
    HuggingFace's ``SamMaskEmbedding`` and produces a feature map
    whose spatial shape matches the image embedding grid so the
    result can be added directly to the image embeddings inside the
    mask decoder.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        inputs: Input mask tensor of shape
            ``(batch, 4*emb, 4*emb, 1)`` (or channels-first).
        hidden_size: int, output embedding dimension. Defaults to ``256``.
        mask_input_channels: int, intermediate channel count. Defaults to ``16``.
        layer_norm_eps: float, epsilon for layer normalization. Defaults to ``1e-6``.
        data_format: string, image data format. Defaults to ``"channels_last"``.
        name: string, name prefix for sub-layers. Defaults to ``"mask_embed"``.

    Returns:
        Dense embedding tensor of shape
        ``(batch, emb, emb, hidden_size)`` (or channels-first).
    """
    cf = data_format == "channels_first"
    inner_channels = mask_input_channels // 4

    x = layers.Conv2D(
        inner_channels,
        kernel_size=2,
        strides=2,
        data_format=data_format,
        name=f"{name}_conv1",
    )(inputs)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln1_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm1")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln1_post_permute")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)

    x = layers.Conv2D(
        mask_input_channels,
        kernel_size=2,
        strides=2,
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln2_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm2")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln2_post_permute")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_2")(x)

    x = layers.Conv2D(
        hidden_size,
        kernel_size=1,
        data_format=data_format,
        name=f"{name}_conv3",
    )(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class SAM(keras.Model):
    """Segment Anything Model (SAM) for promptable image segmentation.

    SAM treats image segmentation as a promptable task, producing
    high-quality masks from flexible user inputs. The architecture
    consists of three components:

    1. **Vision Encoder** – a ViT backbone with windowed attention
       and decomposed relative positional embeddings that produces
       dense image embeddings. A projection neck maps the encoder
       output to the mask decoder hidden dimension.
    2. **Prompt Encoder** – encodes sparse prompts (points, boxes)
       via Fourier positional encoding with learned type embeddings,
       and dense prompts (masks) via a small CNN.
    3. **Mask Decoder** – a lightweight two-way transformer that
       jointly attends between prompt tokens and image embeddings,
       then predicts multiple segmentation masks and corresponding
       IoU quality scores via hypernetwork MLPs.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        vision_hidden_size: Integer, hidden dimension of the vision
            encoder transformer layers. Defaults to ``768``.
        vision_num_hidden_layers: Integer, number of transformer
            layers in the vision encoder. Defaults to ``12``.
        vision_num_attention_heads: Integer, number of attention
            heads per vision encoder layer. Defaults to ``12``.
        vision_mlp_dim: Integer, hidden dimension of the MLP in
            each vision encoder layer. Defaults to ``3072``.
        vision_global_attn_indexes: Tuple of integers, layer
            indices that use global (non-windowed) attention.
            Defaults to ``(2, 5, 8, 11)``.
        num_multimask_outputs: Integer, number of mask outputs
            (excluding the single-mask token).
            Defaults to ``3``.
        multimask_output: Boolean. If ``True``, returns the three
            multimask tokens (indices 1..3); if ``False``, returns
            only the single best-mask token (index 0). Defaults to
            ``True``.
        input_shape: Optional tuple of integers specifying the
            input image shape ``(H, W, C)``. Defaults to
            ``(1024, 1024, 3)``.
        input_tensor: Optional Keras tensor to use as the model
            input.
        name: String, the name of the model.
            Defaults to ``"SAM"``.
        **kwargs: Additional keyword arguments passed to the
            ``keras.Model`` class.

    Inputs:
        - ``pixel_values``: ``(batch, H, W, 3)`` image tensor.
        - ``input_points``: ``(batch, point_batch, num_points, 2)``
          sparse point coordinates in the input image frame.
        - ``input_labels``: ``(batch, point_batch, num_points)``
          point labels (``1``: foreground, ``0``: background,
          ``-1``: not-a-point pad, ``-10``: ignore).
        - ``input_boxes``: ``(batch, point_batch, 4)`` boxes in
          ``(x1, y1, x2, y2)``. Dim-1 must match ``point_batch``
          (one box per prompt slot). Pass zeros when ``has_boxes
          _input`` is 0.
        - ``input_masks``: ``(batch, 4*emb, 4*emb, 1)`` dense
          mask prompt. Pass zeros when not used.
        - ``has_boxes_input``: ``(batch, 1)`` flag — ``1.0`` if
          ``input_boxes`` is meaningful, else ``0.0``. When 0,
          the box slots are populated with the learned
          ``not_a_point`` embedding so the decoder treats them
          as padding.
        - ``has_mask_input``: ``(batch, 1)`` flag — ``1.0`` if
          ``input_masks`` is meaningful, else ``0.0``.

    Returns:
        A ``keras.Model`` instance with dict outputs:
        - ``"pred_masks"``: ``(batch, point_batch, 3|1, H', W')``
          — three masks if ``multimask_output=True``, else one.
        - ``"iou_scores"``: ``(batch, point_batch, 3|1)``.

    Sub-models and helpers:
        The constructor also builds two reusable sub-models on
        ``self.vision_encoder_model`` and
        ``self.prompt_decoder_model``. Use
        :meth:`get_image_embeddings` to run the vision encoder
        alone (cache these to reuse across many prompts), and
        :meth:`get_prompt_embeddings` to return sparse + dense
        prompt embeddings without running the mask decoder.

    Example:
        ```python
        model = kmodels.models.sam.SAM_ViT_Huge(
            input_shape=(1024, 1024, 3),
            weights="sa1b",
        )
        ```
    """

    # Constants shared across all SAM variants
    VISION_OUTPUT_CHANNELS = 256
    VISION_PATCH_SIZE = 16
    VISION_IMAGE_SIZE = 1024
    VISION_WINDOW_SIZE = 14
    VISION_LAYER_NORM_EPS = 1e-6
    MASK_DECODER_HIDDEN_SIZE = 256
    MASK_DECODER_NUM_HIDDEN_LAYERS = 2
    MASK_DECODER_NUM_ATTENTION_HEADS = 8
    MASK_DECODER_MLP_DIM = 2048
    MASK_DECODER_IOU_HEAD_DEPTH = 3
    MASK_DECODER_IOU_HEAD_HIDDEN_DIM = 256
    PROMPT_ENCODER_HIDDEN_SIZE = 256
    PROMPT_ENCODER_MASK_INPUT_CHANNELS = 16
    PROMPT_ENCODER_NUM_POINT_EMBEDDINGS = 4

    def __init__(
        self,
        vision_hidden_size=768,
        vision_num_hidden_layers=12,
        vision_num_attention_heads=12,
        vision_mlp_dim=3072,
        vision_global_attn_indexes=(2, 5, 8, 11),
        num_multimask_outputs=3,
        multimask_output=True,
        enable_boxes=False,
        enable_masks=False,
        input_shape=None,
        input_tensor=None,
        name="SAM",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()

        if input_shape is None:
            input_shape = (self.VISION_IMAGE_SIZE, self.VISION_IMAGE_SIZE, 3)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        if data_format == "channels_first":
            spatial_size = input_shape[1]
        else:
            spatial_size = input_shape[0]
        image_embedding_size = spatial_size // self.VISION_PATCH_SIZE
        mask_input_size = image_embedding_size * 4

        if data_format == "channels_first":
            image_embed_in_shape = (
                self.VISION_OUTPUT_CHANNELS,
                image_embedding_size,
                image_embedding_size,
            )
            mask_in_shape = (1, mask_input_size, mask_input_size)
        else:
            image_embed_in_shape = (
                image_embedding_size,
                image_embedding_size,
                self.VISION_OUTPUT_CHANNELS,
            )
            mask_in_shape = (mask_input_size, mask_input_size, 1)

        input_points = layers.Input(
            shape=(None, None, 2), name="input_points", dtype="float32"
        )
        input_labels = layers.Input(
            shape=(None, None), name="input_labels", dtype="int32"
        )

        input_boxes = None
        has_boxes_input = None
        if enable_boxes:
            input_boxes = layers.Input(
                shape=(None, 4), name="input_boxes", dtype="float32"
            )
            has_boxes_input = layers.Input(
                shape=(1,), name="has_boxes_input", dtype="float32"
            )

        input_masks = None
        has_mask_input = None
        input_mask_embedding = None
        if enable_masks:
            input_masks = layers.Input(
                shape=mask_in_shape, name="input_masks", dtype="float32"
            )
            has_mask_input = layers.Input(
                shape=(1,), name="has_mask_input", dtype="float32"
            )

        hidden_states = layers.Conv2D(
            vision_hidden_size,
            kernel_size=self.VISION_PATCH_SIZE,
            strides=self.VISION_PATCH_SIZE,
            padding="valid",
            use_bias=True,
            data_format=data_format,
            name="vision_encoder_patch_embed_projection",
        )(pixel_values)

        pos_embed_layer = SAMAbsolutePositionEmbedding(
            vision_hidden_size,
            image_embedding_size,
            data_format=data_format,
            name="vision_encoder_pos_embed",
        )
        hidden_states = pos_embed_layer(hidden_states)

        for i in range(vision_num_hidden_layers):
            win_size = (
                self.VISION_WINDOW_SIZE if i not in vision_global_attn_indexes else 0
            )
            hidden_states = SAMVisionLayer(
                vision_hidden_size,
                vision_num_attention_heads,
                vision_mlp_dim,
                qkv_bias=True,
                use_rel_pos=True,
                window_size=win_size,
                image_size=image_embedding_size,
                layer_norm_eps=self.VISION_LAYER_NORM_EPS,
                data_format=data_format,
                name=f"vision_encoder_layers_{i}",
            )(hidden_states)

        image_embeddings = sam_vision_neck(
            hidden_states,
            self.VISION_OUTPUT_CHANNELS,
            data_format=data_format,
            name="vision_encoder_neck",
        )

        num_pos_feats = 128
        shared_image_embedding = SAMPositionalEmbedding(
            num_pos_feats=num_pos_feats,
            scale=vision_hidden_size // 2,
            name="shared_image_embedding",
        )

        image_pe_layer = SAMImagePositionalEmbeddings(
            image_embedding_size,
            shared_image_embedding,
            name="image_positional_embeddings",
        )
        image_pe = image_pe_layer(image_embeddings)

        if enable_masks:
            input_mask_embedding = sam_mask_embedding(
                input_masks,
                hidden_size=self.PROMPT_ENCODER_HIDDEN_SIZE,
                mask_input_channels=self.PROMPT_ENCODER_MASK_INPUT_CHANNELS,
                layer_norm_eps=self.VISION_LAYER_NORM_EPS,
                data_format=data_format,
                name="prompt_encoder_mask_embed",
            )

        prompt_encoder_layer = SAMPromptEncoderLayer(
            hidden_size=self.PROMPT_ENCODER_HIDDEN_SIZE,
            image_embedding_size=image_embedding_size,
            image_size=self.VISION_IMAGE_SIZE,
            num_point_embeddings=self.PROMPT_ENCODER_NUM_POINT_EMBEDDINGS,
            shared_embedding=shared_image_embedding,
            enable_boxes=enable_boxes,
            enable_masks=enable_masks,
            data_format=data_format,
            name="prompt_encoder",
        )

        def _prompt_inputs_dict():
            d = {"input_points": input_points, "input_labels": input_labels}
            if enable_boxes:
                d["input_boxes"] = input_boxes
                d["has_boxes_input"] = has_boxes_input
            if enable_masks:
                d["input_mask_embedding"] = input_mask_embedding
                d["has_mask_input"] = has_mask_input
            return d

        prompt_results = prompt_encoder_layer(_prompt_inputs_dict())

        sparse_embeddings = prompt_results["sparse_embeddings"]
        dense_embeddings = prompt_results["dense_embeddings"]

        mask_decoder_layer = SAMMaskDecoderLayer(
            hidden_size=self.MASK_DECODER_HIDDEN_SIZE,
            num_hidden_layers=self.MASK_DECODER_NUM_HIDDEN_LAYERS,
            num_attention_heads=self.MASK_DECODER_NUM_ATTENTION_HEADS,
            mlp_dim=self.MASK_DECODER_MLP_DIM,
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=self.MASK_DECODER_IOU_HEAD_DEPTH,
            iou_head_hidden_dim=self.MASK_DECODER_IOU_HEAD_HIDDEN_DIM,
            multimask_output=multimask_output,
            data_format=data_format,
            name="mask_decoder",
        )
        decoder_output = mask_decoder_layer(
            [
                image_embeddings,
                image_pe,
                sparse_embeddings,
                dense_embeddings,
            ]
        )

        pred_masks = decoder_output["pred_masks"]
        iou_scores = decoder_output["iou_scores"]

        main_inputs = {
            "pixel_values": pixel_values,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if enable_boxes:
            main_inputs["input_boxes"] = input_boxes
            main_inputs["has_boxes_input"] = has_boxes_input
        if enable_masks:
            main_inputs["input_masks"] = input_masks
            main_inputs["has_mask_input"] = has_mask_input

        super().__init__(
            inputs=main_inputs,
            outputs={"pred_masks": pred_masks, "iou_scores": iou_scores},
            name=name,
            **kwargs,
        )

        self.vision_hidden_size = vision_hidden_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_global_attn_indexes = list(vision_global_attn_indexes)
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_output = multimask_output
        self.enable_boxes = enable_boxes
        self.enable_masks = enable_masks
        self.image_embedding_size = image_embedding_size
        self.mask_input_size = mask_input_size
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

        self._prompt_encoder_layer = prompt_encoder_layer
        self._mask_decoder_layer = mask_decoder_layer
        self._image_pe_layer = image_pe_layer

        image_embeddings_input = layers.Input(
            shape=image_embed_in_shape,
            name="image_embeddings",
            dtype=pixel_values.dtype,
        )
        decoder_side_image_pe = image_pe_layer(image_embeddings_input)
        decoder_prompt_results = prompt_encoder_layer(_prompt_inputs_dict())
        decoder_side_outputs = mask_decoder_layer(
            [
                image_embeddings_input,
                decoder_side_image_pe,
                decoder_prompt_results["sparse_embeddings"],
                decoder_prompt_results["dense_embeddings"],
            ]
        )

        prompt_decoder_inputs = {
            "image_embeddings": image_embeddings_input,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        prompt_encoder_inputs = {
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if enable_boxes:
            prompt_decoder_inputs["input_boxes"] = input_boxes
            prompt_decoder_inputs["has_boxes_input"] = has_boxes_input
            prompt_encoder_inputs["input_boxes"] = input_boxes
            prompt_encoder_inputs["has_boxes_input"] = has_boxes_input
        if enable_masks:
            prompt_decoder_inputs["input_masks"] = input_masks
            prompt_decoder_inputs["has_mask_input"] = has_mask_input
            prompt_encoder_inputs["input_masks"] = input_masks
            prompt_encoder_inputs["has_mask_input"] = has_mask_input

        self.prompt_decoder_model = keras.Model(
            inputs=prompt_decoder_inputs,
            outputs={
                "pred_masks": decoder_side_outputs["pred_masks"],
                "iou_scores": decoder_side_outputs["iou_scores"],
            },
            name=f"{name}_prompt_decoder",
        )

        self.vision_encoder_model = keras.Model(
            inputs=pixel_values,
            outputs=image_embeddings,
            name=f"{name}_vision_encoder",
        )

        self.prompt_encoder_model = keras.Model(
            inputs=prompt_encoder_inputs,
            outputs={
                "sparse_embeddings": prompt_results["sparse_embeddings"],
                "dense_embeddings": prompt_results["dense_embeddings"],
            },
            name=f"{name}_prompt_encoder_model",
        )

    def get_image_embeddings(self, pixel_values):
        """Run only the vision encoder to produce image embeddings.

        Use this to cache image features once and reuse them across
        many prompt combinations via :attr:`prompt_decoder_model`.

        Args:
            pixel_values: Tensor or array of shape ``(batch, H, W, 3)``
                (or channels-first equivalent).

        Returns:
            Image embeddings of shape
            ``(batch, image_embedding_size, image_embedding_size,
            256)`` (or channels-first equivalent).
        """
        return self.vision_encoder_model(pixel_values)

    def get_prompt_embeddings(
        self,
        input_points,
        input_labels,
        input_boxes=None,
        input_masks=None,
        has_boxes_input=None,
        has_mask_input=None,
    ):
        """Run only the prompt encoder.

        Mirrors HuggingFace ``SamModel.get_prompt_embeddings``. Returns
        the sparse and dense prompt embeddings without invoking the
        mask decoder. Only the arguments that match the model's
        ``enable_boxes`` / ``enable_masks`` configuration are used;
        the rest are ignored.

        Returns:
            Dict with keys ``"sparse_embeddings"`` and
            ``"dense_embeddings"``.
        """
        inputs = {"input_points": input_points, "input_labels": input_labels}
        if self.enable_boxes:
            inputs["input_boxes"] = input_boxes
            inputs["has_boxes_input"] = has_boxes_input
        if self.enable_masks:
            inputs["input_masks"] = input_masks
            inputs["has_mask_input"] = has_mask_input
        return self.prompt_encoder_model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_size": self.vision_hidden_size,
                "vision_num_hidden_layers": self.vision_num_hidden_layers,
                "vision_num_attention_heads": self.vision_num_attention_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_global_attn_indexes": self.vision_global_attn_indexes,
                "num_multimask_outputs": self.num_multimask_outputs,
                "multimask_output": self.multimask_output,
                "enable_boxes": self.enable_boxes,
                "enable_masks": self.enable_masks,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_sam_model(
    variant,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    """Factory function for creating SAM model variants.

    Looks up the architecture configuration for the given variant
    name, instantiates a ``SAM`` model, and optionally loads
    pretrained weights from the configured URL or a local file
    path.

    Args:
        variant: String, model variant name (e.g.,
            ``"SAM_ViT_Huge"``).
        input_shape: Optional tuple of integers specifying the
            input shape ``(H, W, C)``. Defaults to
            ``(1024, 1024, 3)``.
        input_tensor: Optional Keras tensor to use as the model
            input.
        weights: String, one of ``None`` (random initialization),
            a weight identifier from the config (e.g.,
            ``"sa1b"``), or a path to a weights file.
        **kwargs: Additional keyword arguments passed to the
            ``SAM`` constructor.

    Returns:
        A configured ``SAM`` model instance.
    """
    config = SAM_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if input_shape is None:
        image_size = SAM.VISION_IMAGE_SIZE
        df = keras.config.image_data_format()
        if df == "channels_first":
            input_shape = (3, image_size, image_size)
        else:
            input_shape = (image_size, image_size, 3)
        print(f"Using default input shape {input_shape}.")

    model = SAM(
        vision_hidden_size=config["vision_hidden_size"],
        vision_num_hidden_layers=config["vision_num_hidden_layers"],
        vision_num_attention_heads=config["vision_num_attention_heads"],
        vision_mlp_dim=config["vision_mlp_dim"],
        vision_global_attn_indexes=config["vision_global_attn_indexes"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        print(f"Loading {weights} weights for {variant}.")
        load_weights_from_config(variant, weights, model, SAM_WEIGHTS_CONFIG)
    elif weights is not None and isinstance(weights, str):
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def SAM_ViT_Base(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Base",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SAM_ViT_Large(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Large",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )


@register_model
def SAM_ViT_Huge(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam_model(
        "SAM_ViT_Huge",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
