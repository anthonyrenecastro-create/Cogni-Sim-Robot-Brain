import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    payload: Dict[str, Any] = {
        "ts_utc": _iso_now(),
        "level": level,
        "event": event,
    }
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True), flush=True)


def log_failure(event: str, exc: Exception, **fields: Any) -> None:
    log_event(
        event,
        level="error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        **fields,
    )


class MetricsTimer:
    def __init__(self):
        self._t0 = time.perf_counter()

    def elapsed_sec(self) -> float:
        return time.perf_counter() - self._t0


def fail_exit(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)
