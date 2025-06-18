import keras
from keras import layers, ops, initializers

@keras.saving.register_keras_serializable(package="kvmm")
class SigLIPAttention(keras.Layer):
    """Multi-head attention layer for SigLip model.
    
    This layer implements scaled dot-product multi-head attention with optional
    combined query-key-value projection for efficiency. It supports both self-attention
    and cross-attention patterns.
    
    Args:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of each attention head.
        attention_dropout (float, optional): Dropout rate for attention weights. 
            Defaults to 0.0.
        use_bias (bool, optional): Whether to use bias in linear projections. 
            Defaults to True.
        combined_qkv (bool, optional): Whether to use a single combined projection
            for query, key, and value instead of separate projections. Can improve
            efficiency. Defaults to False.
        block_prefix (str, optional): Prefix for layer names. Defaults to 
            "multi_head_attention".
        **kwargs: Additional keyword arguments passed to the parent Layer class.
    
    Attributes:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of each attention head.
        dim (int): Total dimension (num_heads * hidden_dim).
        scale (float): Scaling factor for attention scores (1/sqrt(hidden_dim)).
    
    Returns:
        Tensor: Attention output of shape (batch_size, seq_len, dim).
    
    Example:
        >>> attention = SigLipAttention(
        ...     num_heads=8,
        ...     hidden_dim=64,
        ...     attention_dropout=0.1
        ... )
        >>> output = attention(inputs)  # Self-attention
        >>> output = attention(query, key=key, value=value)  # Cross-attention
    """

    def __init__(self,
                 num_heads,
                 hidden_dim,
                 attention_dropout=0.0,
                 use_bias=True,
                 combined_qkv=False,
                 block_prefix="multi_head_attention",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.combined_qkv = combined_qkv
        self.block_prefix = block_prefix

        self.dim = num_heads * hidden_dim
        self.scale = hidden_dim ** -0.5

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        if combined_qkv:
            self.in_proj = layers.Dense(
                3 * self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix+"in_proj"
            )
        else:
            self.q_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix+"q_proj"
            )
            self.k_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix+"k_proj"
            )
            self.v_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix+"v_proj"
            )

        self.out_proj = layers.Dense(
            self.dim,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name=prefix+"out_proj"
        )

        if attention_dropout > 0.0:
            self.dropout = layers.Dropout(attention_dropout, dtype=self.dtype_policy,)
        else:
            self.dropout = None

    def call(self, inputs, key=None, value=None, training=None):
        if key is None:
            key = inputs
        if value is None:
            value = inputs

        batch_size = ops.shape(inputs)[0]

        if self.combined_qkv:
            q_proj = self.in_proj(inputs)
            k_proj = self.in_proj(key)
            v_proj = self.in_proj(value)

            q = q_proj[..., :self.dim]
            k = k_proj[..., self.dim:2*self.dim]
            v = v_proj[..., 2*self.dim:]
        else:
            q = self.q_proj(inputs)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = ops.reshape(q, (batch_size, -1, self.num_heads, self.hidden_dim))
        k = ops.reshape(k, (batch_size, -1, self.num_heads, self.hidden_dim))
        v = ops.reshape(v, (batch_size, -1, self.num_heads, self.hidden_dim))

        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        attn_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_scores, axis=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, (batch_size, -1, self.dim))
        output = self.out_proj(attn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "attention_dropout": self.attention_dropout,
            "use_bias": self.use_bias,
            "combined_qkv": self.combined_qkv,
        })
        return config

@keras.saving.register_keras_serializable(package="kvmm")
class Probe(layers.Layer):
    """Learnable probe parameter layer.
    
    This layer creates a learnable parameter that can be used as a probe token
    or query vector in attention mechanisms. The probe is initialized using
    Glorot uniform initialization and is repeated across the batch dimension
    during the forward pass.
    
    Args:
        hidden_dim (int): Dimension of the probe vector.
        **kwargs: Additional keyword arguments passed to the parent Layer class.
    
    Attributes:
        hidden_dim (int): Dimension of the probe vector.
        probe (Variable): Learnable probe parameter of shape (1, 1, hidden_dim).
    
    Returns:
        Tensor: Probe tensor repeated for each batch element, 
               shape (batch_size, 1, hidden_dim).
    
    Example:
        >>> probe_layer = Probe(hidden_dim=512)
        >>> probe_tokens = probe_layer(inputs)  # Shape: (batch_size, 1, 512)
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.probe = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=initializers.GlorotUniform(),
            dtype=self.dtype_policy.variable_dtype,
            name="probe"
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.repeat(self.probe, repeats=batch_size, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

@keras.saving.register_keras_serializable(package="kvmm")
class PositionIDs(layers.Layer):
    """Position ID generation layer for text and vision inputs.
    
    This layer generates position IDs as a sequence of integers from 0 to max_length-1.
    The position IDs are stored as a non-trainable weight and can be used for
    positional encoding in transformer models.
    
    Args:
        max_length (int): Maximum sequence length for position IDs.
        **kwargs: Additional keyword arguments passed to the parent Layer class.
    
    Attributes:
        max_length (int): Maximum sequence length.
        position_ids (Variable): Non-trainable position ID tensor of shape 
                                (1, max_length).
    
    Returns:
        Tensor: Position IDs tensor of shape (1, max_length).
    
    Example:
        >>> pos_layer = PositionIDs(max_length=512)
        >>> position_ids = pos_layer(inputs)  # Shape: (1, 512), values [0, 1, ..., 511]
    """

    def __init__(self, max_length, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def build(self, input_shape):
        self.position_ids = self.add_weight(
            shape=(1, self.max_length),
            initializer="zeros",
            dtype="int32",
            trainable=False,
            name="position_ids",
        )
        self.position_ids.assign(
            ops.expand_dims(ops.arange(0, self.max_length), axis=0)
        )

    def call(self, inputs):
        return self.position_ids

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length})
        return config

@keras.saving.register_keras_serializable(package="kvmm")
class LogitScaleBias(layers.Layer):
    """Learnable logit scaling and bias layer for contrastive learning.
    
    This layer applies learnable scaling and bias to similarity matrices, commonly
    used in contrastive learning frameworks like CLIP and SigLip. The scaling
    parameter is initialized as log(1.0) and the bias as zero.
    
    Attributes:
        logit_scale (Variable): Learnable scaling parameter (stored as log value).
        logit_bias (Variable): Learnable bias parameter.
    
    Returns:
        Tensor: Scaled and biased similarity matrix of the same shape as input.
    
    Example:
        >>> scale_bias = LogitScaleBias()
        >>> scaled_logits = scale_bias(similarity_matrix)
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=initializers.Constant(ops.log(1.0)),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )
        self.logit_bias = self.add_weight(
            shape=(),
            initializer=initializers.Zeros(),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_bias",
        )

    def call(self, similarity_matrix):
        scaled_logits = ops.multiply(similarity_matrix, ops.exp(self.logit_scale))
        return ops.add(scaled_logits, self.logit_bias)