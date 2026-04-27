"""On-the-fly HF → Keras weight conversion for Qwen2-VL.

Mirrors the pattern used by MetaCLIP 2 / SAM3:

  * First call: downloads HF checkpoint, runs the transfer function,
    writes cache under ``~/.cache/kmodels/<model_name>/``.
  * Subsequent calls: instant load from cache.

Because ``Qwen2VL2B`` returns a *bundle* of three sub-models
(``embed_tokens``, ``llm``, ``vision``) rather than a single model, this
module exposes its own ``load_and_convert_bundle_from_hf`` helper that
saves/loads them as three separate files in the cache directory.
"""

import os
from typing import Callable, Dict

import keras
import numpy as np

# ----------------------------------------------------------------------------
# Weight mapping — reused from the standalone converter, but now consumes a
# pre-built ``{key: ndarray}`` state dict rather than opening safetensors.
# ----------------------------------------------------------------------------


def _t2np(t):
    # Accepts torch tensors (bf16/fp16 → fp32) or numpy arrays.
    try:
        import torch

        if isinstance(t, torch.Tensor):
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(torch.float32)
            return t.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(t)


def _build_kmap(
    state_dict: Dict[str, np.ndarray], cfg: dict
) -> Dict[str, Dict[str, np.ndarray]]:
    tc = cfg["text_config"]
    vc = cfg["vision_config"]
    n_layers = tc["num_hidden_layers"]
    v_depth = vc["depth"]
    kmap: Dict[str, Dict[str, np.ndarray]] = {}

    def put(layer, var, arr):
        kmap.setdefault(layer, {})[var] = arr

    def g(name):
        return _t2np(state_dict[name])

    # Detect state-dict layout: ``from_pretrained`` yields
    # ``model.language_model.*`` + ``model.visual.*`` + ``lm_head.weight``,
    # while raw safetensors use ``model.*`` + ``visual.*``.
    if "model.language_model.embed_tokens.weight" in state_dict:
        llm_pfx = "model.language_model"
        vis_pfx = "model.visual"
    else:
        llm_pfx = "model"
        vis_pfx = "visual"

    put("model_embed_tokens", "embeddings", g(f"{llm_pfx}.embed_tokens.weight"))

    for i in range(n_layers):
        base = f"{llm_pfx}.layers.{i}"
        kp = f"model_layers_{i}"
        put(f"{kp}_input_layernorm", "weight", g(f"{base}.input_layernorm.weight"))
        for proj in ("q_proj", "k_proj", "v_proj"):
            w = g(f"{base}.self_attn.{proj}.weight")
            b = g(f"{base}.self_attn.{proj}.bias")
            put(f"{kp}_self_attn_{proj}", "kernel", np.transpose(w, (1, 0)))
            put(f"{kp}_self_attn_{proj}", "bias", b)
        o_w = g(f"{base}.self_attn.o_proj.weight")
        put(f"{kp}_self_attn_o_proj", "kernel", np.transpose(o_w, (1, 0)))
        put(
            f"{kp}_post_attention_layernorm",
            "weight",
            g(f"{base}.post_attention_layernorm.weight"),
        )
        for gp in ("gate_proj", "up_proj", "down_proj"):
            w = g(f"{base}.mlp.{gp}.weight")
            put(f"{kp}_mlp_{gp}", "kernel", np.transpose(w, (1, 0)))

    put("model_norm", "weight", g(f"{llm_pfx}.norm.weight"))

    # Untied LM head (7B / 72B). HF stores ``lm_head.weight`` at the top
    # level alongside the wrapped ``model.*`` keys.
    if not tc["tie_word_embeddings"]:
        if "lm_head.weight" in state_dict:
            lm_head_w = g("lm_head.weight")
            put("lm_head", "kernel", np.transpose(lm_head_w, (1, 0)))
        else:
            raise KeyError(
                "tie_word_embeddings=False but 'lm_head.weight' missing from "
                "the HF state dict."
            )

    w = g(f"{vis_pfx}.patch_embed.proj.weight")
    w = w.reshape(w.shape[0], -1)
    put("visual_patch_embed_proj", "kernel", np.transpose(w, (1, 0)))

    for i in range(v_depth):
        base = f"{vis_pfx}.blocks.{i}"
        kp = f"visual_blocks_{i}"
        put(f"{kp}_norm1", "gamma", g(f"{base}.norm1.weight"))
        put(f"{kp}_norm1", "beta", g(f"{base}.norm1.bias"))
        qkv_w = g(f"{base}.attn.qkv.weight")
        put(f"{kp}_attn_qkv", "kernel", np.transpose(qkv_w, (1, 0)))
        put(f"{kp}_attn_qkv", "bias", g(f"{base}.attn.qkv.bias"))
        proj_w = g(f"{base}.attn.proj.weight")
        put(f"{kp}_attn_proj", "kernel", np.transpose(proj_w, (1, 0)))
        put(f"{kp}_attn_proj", "bias", g(f"{base}.attn.proj.bias"))
        put(f"{kp}_norm2", "gamma", g(f"{base}.norm2.weight"))
        put(f"{kp}_norm2", "beta", g(f"{base}.norm2.bias"))
        fc1_w = g(f"{base}.mlp.fc1.weight")
        put(f"{kp}_mlp_fc1", "kernel", np.transpose(fc1_w, (1, 0)))
        put(f"{kp}_mlp_fc1", "bias", g(f"{base}.mlp.fc1.bias"))
        fc2_w = g(f"{base}.mlp.fc2.weight")
        put(f"{kp}_mlp_fc2", "kernel", np.transpose(fc2_w, (1, 0)))
        put(f"{kp}_mlp_fc2", "bias", g(f"{base}.mlp.fc2.bias"))

    put("visual_merger_ln_q", "gamma", g(f"{vis_pfx}.merger.ln_q.weight"))
    put("visual_merger_ln_q", "beta", g(f"{vis_pfx}.merger.ln_q.bias"))
    m0_w = g(f"{vis_pfx}.merger.mlp.0.weight")
    put("visual_merger_mlp_0", "kernel", np.transpose(m0_w, (1, 0)))
    put("visual_merger_mlp_0", "bias", g(f"{vis_pfx}.merger.mlp.0.bias"))
    m2_w = g(f"{vis_pfx}.merger.mlp.2.weight")
    put("visual_merger_mlp_2", "kernel", np.transpose(m2_w, (1, 0)))
    put("visual_merger_mlp_2", "bias", g(f"{vis_pfx}.merger.mlp.2.bias"))

    return kmap


def _iter_all_layers(root):
    seen = set()

    def _walk(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        yield layer
        for attr in vars(layer).values():
            if isinstance(attr, keras.layers.Layer):
                yield from _walk(attr)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, keras.layers.Layer):
                        yield from _walk(item)

    if hasattr(root, "layers"):
        for top in root.layers:
            yield from _walk(top)
    else:
        yield from _walk(root)


def _assign(model_or_layer, kmap):
    assigned = 0
    for layer in _iter_all_layers(model_or_layer):
        if layer.name not in kmap:
            continue
        layer_weights = kmap[layer.name]
        new_values = []
        for w in layer.weights:
            short = w.path.rsplit("/", 1)[-1].split(":")[0]
            if short in layer_weights:
                arr = layer_weights[short]
                if tuple(arr.shape) != tuple(w.shape):
                    raise ValueError(
                        f"Shape mismatch for {layer.name}/{short}: got {arr.shape} "
                        f"expected {tuple(w.shape)}"
                    )
                new_values.append(arr)
                assigned += 1
            else:
                new_values.append(w.numpy())
        if new_values:
            layer.set_weights(new_values)
    return assigned


def transfer_qwen2_vl_weights(
    bundle: dict, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Apply an HF Qwen2-VL state dict to a kmodels bundle in-place."""
    kmap = _build_kmap(hf_state_dict, bundle["config"])
    _assign(bundle["embed_tokens"], kmap)
    _assign(bundle["llm"], kmap)
    _assign(bundle["vision"], kmap)
    if bundle.get("lm_head") is not None:
        # Build the Dense once we know the input feature size.
        bundle["lm_head"].build((None, bundle["config"]["text_config"]["hidden_size"]))
        _assign(bundle["lm_head"], kmap)


# ----------------------------------------------------------------------------
# Bundle-aware downloader / cache
# ----------------------------------------------------------------------------


def _cache_dir(model_name: str) -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "kmodels", model_name)


def _paths(model_name: str, *, with_lm_head: bool):
    d = _cache_dir(model_name)
    paths = {
        "embed": os.path.join(d, f"{model_name}_embed.weights.h5"),
        "llm": os.path.join(d, f"{model_name}_llm.weights.h5"),
        "vision": os.path.join(d, f"{model_name}_vision.weights.h5"),
    }
    if with_lm_head:
        paths["lm_head"] = os.path.join(d, f"{model_name}_lm_head.weights.h5")
    return paths


def load_and_convert_bundle_from_hf(
    bundle: dict,
    model_name: str,
    hf_model_id: str,
    transfer_fn: Callable[[dict, Dict[str, np.ndarray]], None],
    hf_model_cls: str = None,
    hf_kwargs: dict = None,
) -> None:
    """Download HF weights for a Qwen2-VL-style bundle; cache split files.

    If the cache files already exist, just load them and return.
    Otherwise: pull from HF, run ``transfer_fn`` on the state dict, save
    three split weight files under ``~/.cache/kmodels/<model_name>/``.
    """
    has_lm_head = bundle.get("lm_head") is not None
    paths = _paths(model_name, with_lm_head=has_lm_head)
    if all(os.path.exists(p) for p in paths.values()):
        print(f"Loading cached {model_name} weights from {_cache_dir(model_name)}")
        # Wrap embed_tokens in a Sequential so save/load_weights has a home.
        emb = keras.Sequential([bundle["embed_tokens"]])
        emb.build((None,))
        emb.load_weights(paths["embed"])
        bundle["llm"].load_weights(paths["llm"])
        bundle["vision"].load_weights(paths["vision"])
        if has_lm_head:
            hidden = bundle["config"]["text_config"]["hidden_size"]
            lm_head_wrapper = keras.Sequential([bundle["lm_head"]])
            lm_head_wrapper.build((None, hidden))
            lm_head_wrapper.load_weights(paths["lm_head"])
        return

    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as e:
        raise ImportError(
            f"Converting {model_name} weights requires `torch` and `transformers`. "
            "Install with: pip install torch transformers"
        ) from e

    print(f"Downloading {model_name} from HuggingFace ({hf_model_id})...")
    hf_token = os.environ.get("HF_TOKEN")
    kwargs = {"token": hf_token} if hf_token else {}
    if hf_kwargs:
        kwargs.update(hf_kwargs)

    cls_name = hf_model_cls or "Qwen2VLForConditionalGeneration"
    cls = getattr(transformers, cls_name)
    hf_model = cls.from_pretrained(hf_model_id, **kwargs).eval()

    # Build a plain dict {name: torch.Tensor}; transfer_fn does the dtype cast.
    hf_state_dict = dict(hf_model.state_dict())
    del hf_model

    print(f"Converting {model_name} weights to Keras...")
    bundle["embed_tokens"].build((None,))
    transfer_fn(bundle, hf_state_dict)

    os.makedirs(_cache_dir(model_name), exist_ok=True)
    emb_model = keras.Sequential([bundle["embed_tokens"]])
    emb_model.build((None,))
    emb_model.save_weights(paths["embed"])
    bundle["llm"].save_weights(paths["llm"])
    bundle["vision"].save_weights(paths["vision"])
    if has_lm_head:
        hidden = bundle["config"]["text_config"]["hidden_size"]
        lm_head_wrapper = keras.Sequential([bundle["lm_head"]])
        lm_head_wrapper.build((None, hidden))
        lm_head_wrapper.save_weights(paths["lm_head"])

    submodels = [bundle["embed_tokens"], bundle["llm"], bundle["vision"]]
    if has_lm_head:
        submodels.append(bundle["lm_head"])
    total = sum(sum(np.prod(w.shape) * 4 for w in m.weights) for m in submodels) / (
        1024**3
    )
    print(f"Cached {model_name} weights to {_cache_dir(model_name)} ({total:.1f} GB)")
