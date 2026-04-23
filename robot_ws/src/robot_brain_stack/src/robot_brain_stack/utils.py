import json
import math
import time
from typing import Dict, Any


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def now() -> float:
    return time.time()


def dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(',', ':'))