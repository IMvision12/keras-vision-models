import sys
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# Global registry storage
_MODEL_REGISTRY: List[Dict[str, Any]] = []


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Model registration decorator to track the model name and available weights.

    This decorator registers a model function in a global model registry (`_MODEL_REGISTRY`)
    and ensures that the model name is added to the module's `__all__` list. It also checks
    for available weight configurations and updates the model's registry entry accordingly.

    Args:
    -----------
    fn : Callable[..., Any]
        The model function to be registered.

    Returns:
    --------
    Callable[..., Any]
        The wrapped model function with registration functionality.

    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    mod = sys.modules[fn.__module__]
    model_name = fn.__name__

    if hasattr(mod, "__all__"):
        if model_name not in mod.__all__:
            mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    weight_configs = {}
    for item in dir(mod):
        if isinstance(item, str) and item.endswith("_WEIGHTS_CONFIG"):
            weight_attr = getattr(mod, item)
            if model_name in weight_attr:
                weight_configs = weight_attr[model_name]

    for model in _MODEL_REGISTRY:
        if model["name"] == model_name:
            warnings.warn(
                f"Model '{model_name}' is already registered. Updating registration."
            )
            model["weights"] = weight_configs
            return wrapper

    _MODEL_REGISTRY.append({"name": model_name, "weights": weight_configs.keys()})

    return wrapper


def list_models(pattern: Optional[str] = None) -> None:
    """
    List registered models and their available weights, optionally filtering by a pattern.

    This function displays the names of registered models along with their associated
    weights, with an optional case-insensitive filter to match model names based on a
    specified pattern.

    Args:
    -----------
    pattern : Optional[str], default=None
        A case-insensitive substring to filter model names.
        If provided, only models containing this substring in their names will be listed.

    Returns:
    --------
        This function prints the list of models and weights directly to the console.

    Example Usage:
    --------------
    >>> list_models("convmixer")  # Lists all models with "convmixer" in the name.
    >>> list_models()             # Lists all registered models and their weights.

    """
    if pattern is None:
        filtered_models = _MODEL_REGISTRY
    else:
        pattern = pattern.lower()
        filtered_models = [
            model for model in _MODEL_REGISTRY if pattern in model["name"].lower()
        ]

    if not filtered_models:
        print(f"No models found{f' matching pattern: {pattern}' if pattern else ''}")
        return

    processed_models = {}
    for model in filtered_models:
        if model["weights"]:
            processed_models[model["name"]] = sorted(model["weights"])

    for model_name, weights in sorted(processed_models.items()):
        if len(weights) == 1:
            print(f"{model_name} : {weights[0]}")
        else:
            print(f"{model_name} : {', '.join(weights)}")
