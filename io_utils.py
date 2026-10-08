"""io_utils - endpoints, HTTP access and the local download cache for the public C-Stroke platform.

All data used by slide_utils, pathology_utils and refspace_utils come from the C-Stroke deployment at
https://c-stroke.humanbrain.in (override with the CSTROKE_BASE_URL environment variable):

    sets_data.json                                    slide catalogue (sets of adjacent sections)
    fcgi-bin/iipsrv.fcgi                              IIPImage server for the whole-slide images
    histology_detections/openAtlasJson/...json        pathology annotations (GeoJSON)
    Stroke_BFIW/<block>/...jpg                        block-face photographs
    stroke_atlas/viewer/data/manifest.json, *.bin     MRI reference space, label volumes, block geometry

Downloads are cached under ~/.cache/cstroke (override with CSTROKE_CACHE), so repeated requests are served from
disk. Delete that folder to force fresh downloads.
"""

import os
import re
import time
from pathlib import Path

__all__ = ["BASE_URL", "IIP_URL", "SETS_URL", "BLOCKFACE_URL", "REFSPACE_URL", "CACHE_DIR",
           "get", "get_bytes", "safe_name"]

BASE_URL = os.environ.get("CSTROKE_BASE_URL", "https://c-stroke.humanbrain.in").rstrip("/")
IIP_URL = f"{BASE_URL}/fcgi-bin/iipsrv.fcgi"
SETS_URL = f"{BASE_URL}/sets_data.json"
BLOCKFACE_URL = f"{BASE_URL}/Stroke_BFIW"
REFSPACE_URL = f"{BASE_URL}/stroke_atlas/viewer/data"
CACHE_DIR = Path(os.environ.get("CSTROKE_CACHE", Path.home() / ".cache" / "cstroke"))

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        _SESSION = requests.Session()
        _SESSION.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=16))
        _SESSION.headers["User-Agent"] = "cstroke-notebooks/1.0"
    return _SESSION


def get(url, timeout=180, retries=3):
    """HTTP GET with retries and exponential back-off. Returns the requests.Response."""
    err = None
    for attempt in range(retries):
        try:
            response = _session().get(url, timeout=timeout)
        except Exception as e:  # connection problems are retried
            err = e
        else:
            if response.status_code < 400:
                return response
            if response.status_code < 500:       # missing or forbidden: retrying will not help
                raise FileNotFoundError(f"HTTP {response.status_code}: {url}")
            err = f"HTTP {response.status_code}"
        time.sleep(1.5 * 2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{err}")


def get_bytes(url, cache_name=None, use_cache=True, validate=None):
    """Download `url` (or read it from the cache) and return its content as bytes.

    cache_name  relative path inside CACHE_DIR; None disables caching for this request
    validate    optional callable(bytes) -> bool; content that fails it is not cached and raises ValueError
    """
    path = CACHE_DIR / cache_name if cache_name else None
    if use_cache and path is not None and path.exists():
        return path.read_bytes()
    data = get(url).content
    if validate is not None and not validate(data):
        raise ValueError(f"Unexpected content from {url}")
    if use_cache and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def safe_name(text):
    """File-system safe version of a path or URL fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
