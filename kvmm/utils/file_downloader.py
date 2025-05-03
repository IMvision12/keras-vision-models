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


def download_file(
    file_url: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> str:
    """Download file from the specified URL.
    Args:
        file_url: URL to download file from
        cache_dir: Directory to cache files (default: ~/.downloads)
        force_download: Force download even if file exists
    Returns:
        str: Path to the downloaded file
    Raises:
        ValueError: For invalid inputs
        Exception: For download failures
    """
    if not file_url:
        raise ValueError("file_url cannot be empty")
    if not validate_url(file_url):
        raise ValueError(f"Invalid URL format: {file_url}")

    cache_dir = Path(cache_dir or os.path.expanduser("~/.downloads"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_name = os.path.basename(file_url)
    local_file = cache_dir / file_name

    if local_file.exists() and not force_download:
        print(f"Found cached file at {local_file}")
        return str(local_file)

    try:
        file_path = utils.get_file(
            fname=file_name,
            origin=file_url,
            cache_dir=str(cache_dir),
            cache_subdir="",
            extract=False,
        )
        return file_path
    except Exception as e:
        print(f"Failed to download file: {str(e)}")
        raise
