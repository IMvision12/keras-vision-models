import importlib
import importlib.util
from pathlib import Path

import pytest
import requests


def _collect_all_urls():
    urls = []
    models_dir = Path(__file__).parent.parent.parent / "kmodels" / "models"

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        config_path = model_dir / "config.py"
        if not config_path.exists():
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"config_{model_dir.name}", str(config_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            continue

        for attr_name in dir(mod):
            if not attr_name.endswith("_WEIGHTS_CONFIG"):
                continue
            config_dict = getattr(mod, attr_name)
            if not isinstance(config_dict, dict):
                continue
            for model_name, weights in config_dict.items():
                if not isinstance(weights, dict):
                    continue
                for weight_name, meta in weights.items():
                    if isinstance(meta, dict) and "url" in meta:
                        urls.append(
                            pytest.param(
                                model_name,
                                weight_name,
                                meta["url"],
                                id=f"{model_name}/{weight_name}",
                            )
                        )
    return urls


ALL_URLS = _collect_all_urls()


@pytest.mark.link_validation
@pytest.mark.slow
@pytest.mark.parametrize("model_family,weight_name,url", ALL_URLS)
def test_weight_url_is_valid_and_downloadable(model_family, weight_name, url):
    head_response = requests.head(url, allow_redirects=True, timeout=30)
    assert head_response.status_code == 200, (
        f"Broken link for {model_family}/{weight_name}: "
        f"{url} → HTTP {head_response.status_code}"
    )

    content_type = head_response.headers.get("Content-Type", "")
    assert "text/html" not in content_type, (
        f"URL returned HTML (likely error page) for "
        f"{model_family}/{weight_name}: {content_type}"
    )

    get_response = requests.get(url, stream=True, timeout=30)
    assert get_response.status_code == 200, (
        f"Download failed for {model_family}/{weight_name}: "
        f"HTTP {get_response.status_code}"
    )
    first_chunk = next(get_response.iter_content(chunk_size=1024), None)
    assert first_chunk is not None and len(first_chunk) > 0, (
        f"Empty download for {model_family}/{weight_name}: {url}"
    )
    get_response.close()
