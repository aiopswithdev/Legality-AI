import time
import threading
from typing import Dict, Any, Optional
import uuid

# ---------------------------------------
# Config
# ---------------------------------------

DEFAULT_TTL_SECONDS = 15 * 60  # 15 minutes

# ---------------------------------------
# In-memory cache
# ---------------------------------------

_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# ---------------------------------------
# Public API
# ---------------------------------------

def create_analysis_entry(
    pdf_bytes: bytes,
    analysis_result: Dict[str, Any],
    ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    """
    Store analysis result and return analysis_id.
    """
    analysis_id = str(uuid.uuid4())
    expires_at = time.time() + ttl_seconds

    with _LOCK:
        _ANALYSIS_CACHE[analysis_id] = {
            "pdf_bytes": pdf_bytes,
            "result": analysis_result,
            "expires_at": expires_at
        }

    return analysis_id


def get_analysis_entry(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached analysis if valid.
    """
    now = time.time()

    with _LOCK:
        entry = _ANALYSIS_CACHE.get(analysis_id)
        if not entry:
            return None

        if entry["expires_at"] < now:
            # expired → cleanup
            del _ANALYSIS_CACHE[analysis_id]
            return None

        return entry


def cleanup_expired_entries() -> None:
    """
    Optional manual cleanup (can be called periodically).
    """
    now = time.time()

    with _LOCK:
        expired_keys = [
            k for k, v in _ANALYSIS_CACHE.items()
            if v["expires_at"] < now
        ]
        for k in expired_keys:
            del _ANALYSIS_CACHE[k]
