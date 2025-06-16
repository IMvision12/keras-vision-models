import keras
from keras import ops

from kvmm.layers import AddPositionEmbs


@keras.saving.register_keras_serializable(package="kvmm")
class CLIPAttention(keras.layers.Layer):
    """Multi-head attention layer for CLIP model implementing scaled dot-product attention.

    This layer implements the multi-head attention mechanism used in the CLIP architecture.
    It projects input tensors into query, key, and value representations, applies
    scaled dot-product attention, and projects the output back to the original dimension.

    Key Features:
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with optional attention masking
        - Separate projection matrices for query, key, and value transformations
        - Customizable projection dimensions and number of attention heads
        - Support for sequential inputs with variable sequence lengths

    Args:
        proj_dim (int): Dimension of the projection space. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on proj_dim/num_heads features
        name_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - hidden_states: Tensor of shape (batch_size, sequence_length, input_dim)
        - attention_mask: Optional tensor for masking certain positions

    Output shape:
        - Tuple containing tensor of shape (batch_size, sequence_length, proj_dim)

    Notes:
        - The projection dimension (proj_dim) must be divisible by num_heads
        - Each attention head processes proj_dim/num_heads features
        - Implements the standard scaled dot-product attention formula:
          Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        - Used in CLIP's text and image encoders for contextual feature extraction
    """

    def __init__(self, proj_dim, num_heads, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert proj_dim % num_heads == 0, "proj_dim should be divisible by num_heads"

        q_proj_name = f"{self.name_prefix}_q_proj" if self.name_prefix else "q_proj"
        k_proj_name = f"{self.name_prefix}_k_proj" if self.name_prefix else "k_proj"
        v_proj_name = f"{self.name_prefix}_v_proj" if self.name_prefix else "v_proj"
        out_proj_name = (
            f"{self.name_prefix}_out_proj" if self.name_prefix else "out_proj"
        )

        self.q_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=q_proj_name)
        self.k_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=k_proj_name)
        self.v_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=v_proj_name)
        self.out_proj = keras.layers.Dense(
            self.proj_dim, use_bias=True, name=out_proj_name
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_proj.build((None, input_dim))
        self.k_proj.build((None, input_dim))
        self.v_proj.build((None, input_dim))
        self.out_proj.build((None, self.proj_dim))

        self.built = True

    def transpose_for_scores(self, x):
        batch_size = ops.shape(x)[0]
        seq_length = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_length, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None):
        batch_size = ops.shape(hidden_states)[0]

        x_q = self.q_proj(hidden_states)
        x_k = self.k_proj(hidden_states)
        x_v = self.v_proj(hidden_states)

        x_q = self.transpose_for_scores(x_q)
        x_k = self.transpose_for_scores(x_k)
        x_v = self.transpose_for_scores(x_v)

        x = ops.matmul(x_q, ops.transpose(x_k, (0, 1, 3, 2)))
        x = x * self.scale

        if attention_mask is not None:
            x = x + attention_mask

        x = ops.softmax(x, axis=-1)
        x = ops.matmul(x, x_v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, -1, self.proj_dim))
        x = self.out_proj(x)

        return (x,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kvmm")
class VisionModelEmbedding(keras.layers.Layer):
    """Vision Transformer (ViT) embedding layer that processes image patches.

    This layer follows the Vision Transformer architecture by:
    1. Converting an input image into patches
    2. Adding a special class token embedding (similar to BERT's [CLS] token)
    3. Adding learned positional embeddings to provide spatial information

    The input to this layer should be patch embeddings from an image after
    initial projection to the embedding dimension.

    Args:
        width (int): Dimension of the embedding space.
        input_resolution (int): Resolution of the input image (assumes square images).
        patch_size (int): Size of each image patch (assumes square patches).
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        **kwargs: Additional keyword arguments passed to the parent class.

    Inputs:
        A tensor of shape (batch_size, num_patches, width) representing
        the projected patch embeddings from an image.

    Outputs:
        A tensor of shape (batch_size, num_patches + 1, width) containing
        the patch embeddings plus class token, with positional embeddings added.
    """

    def __init__(
        self, width, input_resolution, patch_size, data_format="channels_last", **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.data_format = data_format
        self.num_patches = (input_resolution // patch_size) ** 2
        self.grid_size = input_resolution // patch_size

        self.position_embs = AddPositionEmbs(
            grid_h=self.grid_size,
            grid_w=self.grid_size,
            no_embed_class=False,
            use_distillation=False,
            name="position_embeddings",
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=((self.width,)),
            name="class_embedding",
        )

        super().build(input_shape)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        if self.data_format == "channels_first":
            patch_embeddings = keras.layers.Reshape((self.width, self.num_patches))(
                inputs
            )
            patch_embeddings = keras.layers.Permute((2, 1))(patch_embeddings)
        else:
            patch_embeddings = keras.layers.Reshape((self.num_patches, self.width))(
                inputs
            )
        class_embed = ops.broadcast_to(
            self.class_embedding, (batch_size, 1, self.width)
        )
        embeddings = ops.concatenate([class_embed, patch_embeddings], axis=1)
        embeddings = self.position_embs(embeddings)

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "input_resolution": self.input_resolution,
                "patch_size": self.patch_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kvmm")
class TextModelEmbedding(keras.layers.Layer):
    """
    A Keras layer that combines token embeddings and positional embeddings for text models.

    This layer is commonly used in transformer-based architectures such as BERT, GPT, and others.
    It performs two key operations:
    1. Converts token IDs to token embeddings using a learned embedding table
    2. Adds positional embeddings to encode position information in the sequence

    The final output is the sum of token embeddings and positional embeddings.

    Args:
        vocab_size (int): Size of the vocabulary, determining the number of unique tokens
        context_length (int): Maximum sequence length to handle
        embedding_dim (int): Dimensionality of the embedding vectors
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        Integer tensor of shape (batch_size, sequence_length) with token IDs

    Output shape:
        Float tensor of shape (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, vocab_size, context_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.token_embedding = keras.layers.Embedding(
            vocab_size, embedding_dim, name="token_embedding"
        )

        self.position_embedding = keras.layers.Embedding(
            context_length, embedding_dim, name="positional_embedding"
        )

    def call(self, inputs):
        token_embeddings = self.token_embedding(inputs)
        batch_size = ops.shape(inputs)[0]
        position_ids = ops.arange(self.context_length, dtype="int32")
        position_ids = ops.expand_dims(position_ids, 0)
        position_embeddings = self.position_embedding(position_ids)
        position_embeddings = ops.tile(position_embeddings, (batch_size, 1, 1))
        return token_embeddings + position_embeddings

    def build(self, input_shape):
        self.token_embedding.build((None, self.context_length))
        self.position_embedding.build((None, self.context_length))
        self.built = True
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.context_length, self.embedding_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kvmm")
class CLIPLogitScale(keras.layers.Layer):
    """
    Learnable temperature parameter for scaling logits in CLIP models.

    This layer implements the learnable temperature parameter used in CLIP to scale
    the dot product similarity between image and text embeddings. The temperature
    is initialized with a value that's typically small (default 0.07) and learned
    during training to improve model convergence.

    Args:
        initial_value (float): Initial temperature value. Default is 0.07.
        **kwargs: Additional keyword arguments passed to the parent class.

    Inputs:
        A tuple of `(image_embeddings, text_embeddings)` where:
        - image_embeddings: Tensor of shape `(batch_size, embed_dim)`
        - text_embeddings: Tensor of shape `(batch_size, embed_dim)`

    Outputs:
        A tuple of `(image_logits, text_logits)` where:
        - image_logits: Tensor of shape `(batch_size, batch_size)`
        - text_logits: Tensor of shape `(batch_size, batch_size)`
    """

    def __init__(self, initial_value=0.07, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        if not isinstance(input_shape, list) and len(input_shape) != 2:
            raise ValueError(
                "CLIPLogitScale expects a list of 2 input shapes (image_embeddings, text_embeddings)"
            )

        self.logit_scale = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(
                value=ops.log(1 / self.initial_value)
            ),
            trainable=True,
            name="logit_scale",
        )

    def call(self, inputs):
        image_embeddings, text_embeddings = inputs
        logit_scale = ops.exp(self.logit_scale)
        image_logits = (
            ops.matmul(image_embeddings, ops.transpose(text_embeddings)) * logit_scale
        )
        text_logits = ops.transpose(image_logits)
        return image_logits, text_logits
