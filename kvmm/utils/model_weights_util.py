from kvmm.utils import download_file


def load_weights_from_config(
    model_name: str, weights_name: str, model, weights_config: dict
):
    """
    Load pre-trained weights for any model architecture.

    Args:
        model_name: Name of the model (e.g., 'EfficientNetB0', 'VGG16', 'ResNet50')
        weights_name: Name of the weights to load (e.g., 'ns_jft_in1k', 'in1k')
        model: The model instance
        weights_config: Dictionary containing weights configuration for the model family

    Returns:
        Model with loaded weights

    Raises:
        ValueError: If model_name or weights_name is invalid
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

    weights_url = model_weights[weights_name]["url"]
    if not weights_url:
        raise ValueError(f"URL for weights '{weights_name}' is not defined")

    try:
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
