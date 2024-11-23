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
