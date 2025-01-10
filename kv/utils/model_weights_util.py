import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from keras import utils


def validate_url(url: str) -> bool:
    """Validate if the provided URL is well-formed.

    Args:
        url: URL string to validate

    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_weights(
    weights_url: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> str:
    """Download model weights from the specified URL.

    Args:
        weights_url: URL to download weights from
        cache_dir: Directory to cache weights (default: ~/.keras/models)
        force_download: Force download even if file exists

    Returns:
        str: Path to the downloaded weights file

    Raises:
        ValueError: For invalid inputs
        Exception: For download failures
    """

    if not weights_url:
        raise ValueError("weights_url cannot be empty")

    if not validate_url(weights_url):
        raise ValueError(f"Invalid URL format: {weights_url}")

    cache_dir = Path(cache_dir or os.path.expanduser("~/.keras/models"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_file = cache_dir / os.path.basename(weights_url)

    if weights_file.exists() and not force_download:
        print(f"Found cached weights at {weights_file}")
        return str(weights_file)

    try:
        weights_path = utils.get_file(
            fname=os.path.basename(weights_url),
            origin=weights_url,
            cache_dir=str(cache_dir),
            cache_subdir="",
            extract=False,
        )

        print("Download complete!")
        return weights_path

    except Exception as e:
        print(f"Failed to download weights: {str(e)}")
        raise


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
        weights_path = download_weights(weights_url)
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
