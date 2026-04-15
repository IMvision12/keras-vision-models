"""On-the-fly weight conversion from HuggingFace gated repos.

For models whose licenses do not allow weight redistribution (e.g. SAM3,
DINOv3), this utility downloads from the original HuggingFace repo,
converts to Keras format using a model-specific transfer function, and
caches the result locally so subsequent loads are instant.

Requires:
    1. User has accepted the model's license on HuggingFace.
    2. ``HF_TOKEN`` env var is set, or ``huggingface-cli login`` has been run.
    3. ``torch`` and ``transformers`` are installed.

Usage::

    from kmodels.weight_utils.hf_weight_converter import load_gated_weights_from_hf

    load_gated_weights_from_hf(
        model=keras_model,
        model_name="sam3",
        hf_model_id="facebook/sam3",
        transfer_fn=transfer_sam3_weights,
        hf_model_cls="Sam3Model",
        hf_kwargs={"attn_implementation": "eager"},
    )
"""

import os


def _get_cache_dir(model_name):
    return os.path.join(os.path.expanduser("~"), ".cache", "kmodels", model_name)


def load_gated_weights_from_hf(
    model,
    model_name,
    hf_model_id,
    transfer_fn,
    hf_model_cls=None,
    hf_kwargs=None,
):
    """Download, convert, and cache HuggingFace weights for a Keras model.

    Weights are cached at ``~/.cache/kmodels/<model_name>/``. Models
    larger than 5 GB are automatically sharded (5 GB per shard).

    Args:
        model: The Keras model instance to load weights into.
        model_name: String used as the cache subdirectory name
            (e.g. ``"sam3"``, ``"dinov3_vitb16"``).
        hf_model_id: HuggingFace model identifier
            (e.g. ``"facebook/sam3"``).
        transfer_fn: Callable ``(keras_model, hf_state_dict) -> None``
            that transfers weights from the HF state dict into the
            Keras model in-place.
        hf_model_cls: Optional string name of the HF model class to
            import from ``transformers`` (e.g. ``"Sam3Model"``).
            If ``None``, uses ``AutoModel``.
        hf_kwargs: Optional dict of extra keyword arguments passed to
            ``from_pretrained`` (e.g. ``{"attn_implementation": "eager"}``).
    """
    cache_dir = _get_cache_dir(model_name)
    cached_weights = os.path.join(cache_dir, f"{model_name}.weights.h5")

    if os.path.exists(cached_weights):
        print(f"Loading cached {model_name} weights from {cached_weights}")
        model.load_weights(cached_weights)
        return

    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as e:
        raise ImportError(
            f"Converting {model_name} weights requires `torch` and `transformers`. "
            "Install them with: pip install torch transformers"
        ) from e

    print(
        f"Downloading {model_name} from HuggingFace "
        f"(requires accepted license + HF token)..."
    )

    hf_token = os.environ.get("HF_TOKEN")
    kwargs = {"token": hf_token}
    if hf_kwargs:
        kwargs.update(hf_kwargs)

    if hf_model_cls is not None:
        cls = getattr(transformers, hf_model_cls)
    else:
        cls = transformers.AutoModel

    try:
        hf_model = cls.from_pretrained(hf_model_id, **kwargs).eval()
    except OSError as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            raise OSError(
                f"\n{'=' * 60}\n"
                f"Access denied for '{hf_model_id}'.\n\n"
                f"This model is gated and requires license acceptance.\n"
                f"Please follow these steps:\n\n"
                f"  1. Go to https://huggingface.co/{hf_model_id}\n"
                f"     and accept the license agreement.\n\n"
                f"  2. Authenticate using one of:\n"
                f"     - Run: huggingface-cli login\n"
                f"     - Or set: export HF_TOKEN=<your_token>\n"
                f"{'=' * 60}"
            ) from e
        raise

    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    del hf_model

    print(f"Converting {model_name} weights to Keras...")
    transfer_fn(model, hf_state_dict)

    os.makedirs(cache_dir, exist_ok=True)

    total_bytes = sum(w.numpy().nbytes for w in model.weights)
    size_gb = total_bytes / (1024**3)
    save_kwargs = {}
    if size_gb > 5:
        save_kwargs["max_shard_size"] = 5.0

    model.save_weights(cached_weights, **save_kwargs)
    print(f"Cached {model_name} weights to {cached_weights} ({size_gb:.1f} GB)")
