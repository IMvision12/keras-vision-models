import json

from kvmm.utils import download_file


def load_weights_from_config(
    model_name: str, weights_name: str, model, weights_config: dict
):
    """
    Load pre-trained weights for any model architecture, with support for
    Keras 3.10+ sharded weights.

    Args:
        model_name: Name of the model (e.g., 'EfficientNetB0', 'VGG16', 'ResNet50')
        weights_name: Name of the weights to load (e.g., 'ns_jft_in1k', 'in1k')
        model: The model instance
        weights_config: Dictionary containing weights configuration for the model family

    Returns:
        Model with loaded weights

    Raises:
        ValueError: If model_name or weights_name is invalid

    Example:
        >>> # Configuration with simplified detection
        >>> weights_config = {
        ...     "ResNet50": {
        ...         "imagenet": {
        ...             "url": "https://example.com/resnet50_imagenet.h5"
        ...         },
        ...         "imagenet_21k": {
        ...             "url": "https://example.com/resnet50_21k.json"  # Sharded weights
        ...         }
        ...     },
        ...     "EfficientNetB0": {
        ...         "imagenet": "https://example.com/efficientnet_b0.h5"
        ...     }
        ... }
        >>>
        >>> # Load single-file weights (.h5, .weights, etc.)
        >>> model = load_weights_from_config("ResNet50", "imagenet", model, weights_config)
        >>>
        >>> # Load sharded weights (automatically detected from .json extension)
        >>> model = load_weights_from_config("ResNet50", "imagenet_21k", model, weights_config)
    """

    if not weights_name or weights_name == "none":
        return model

    if model_name not in weights_config:
        available_models = list(weights_config.keys())
        raise ValueError(
            f"Model '{model_name}' not found in weights config. "
            f"Available models: {available_models}"
        )

    model_weights = weights_config[model_name]
    if weights_name not in model_weights:
        available_weights = list(model_weights.keys())
        raise ValueError(
            f"Weights '{weights_name}' not found for model {model_name}. "
            f"Available weights: {available_weights}"
        )

    weights_info = model_weights[weights_name]

    if isinstance(weights_info, str):
        weights_url = weights_info
    elif isinstance(weights_info, dict) and "url" in weights_info:
        weights_url = weights_info["url"]
    else:
        raise ValueError(f"Invalid weights configuration for '{weights_name}'")

    if not weights_url:
        raise ValueError(f"URL for weights '{weights_name}' is not defined")

    try:
        if weights_url.lower().endswith(".json"):
            json_path = download_file(weights_url)
            with open(json_path, "r") as f:
                json_data = json.load(f)

            base_url = "/".join(weights_url.split("/")[:-1])

            if "weight_map" in json_data:
                weight_files = set(json_data["weight_map"].values())
            else:
                raise ValueError("JSON file must contain either 'weight_map' key")

            for weight_file in sorted(weight_files):
                weight_url = f"{base_url}/{weight_file}"
                _ = download_file(weight_url)

            model.load_weights(json_path)
            return model
        else:
            weights_path = download_file(weights_url)
            model.load_weights(weights_path)
            return model

    except Exception as e:
        raise ValueError(f"Failed to load weights for {model_name}: {str(e)}")


def get_all_weight_names(config: dict) -> list:
    """
    Retrieves all weight names from the given weights configuration dictionary.

    Args:
        config (dict): The weights configuration dictionary.

    Returns:
        list: A list of all weight names.
    """
    weight_names = []
    for model, weights in config.items():
        weight_names.extend(weights.keys())
    return list(set(weight_names))
