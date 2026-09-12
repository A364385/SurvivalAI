"""The single HTTP transport used by every LLM provider.

Owns *how* we speak HTTP — request construction, `urlopen`, JSON decoding — so
no provider implements its own. Error POLICY deliberately stays with each
provider family, because it legitimately differs:

- remote providers (`remote_providers.py`) map failures and retry transient
  ones (429/5xx) behind a circuit breaker, since a hosted API can be briefly
  overloaded;
- local providers (`local_providers.py`) fail fast and wrap the failure in
  `LocalProviderError`, since a local server that is not answering is a
  configuration problem the user can see, not a transient one.

This module therefore raises the underlying `urllib`/JSON errors unchanged and
lets each caller decide what they mean.
"""

import json
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_JSON_HEADERS = {"Content-Type": "application/json"}


def post_json(url: str, payload: Dict[str, Any], *, timeout: float,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """POST a JSON body and decode the JSON response.

    Raises `urllib.error.HTTPError`, `urllib.error.URLError`, `OSError` or
    `json.JSONDecodeError` unchanged so the caller can apply its own policy.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers or dict(DEFAULT_JSON_HEADERS),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, *, timeout: float,
             headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """GET a JSON resource (used for model discovery / health probes)."""
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
