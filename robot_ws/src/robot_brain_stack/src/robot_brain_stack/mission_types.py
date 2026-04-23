from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class Goal:
    goal_id: str
    goal_type: str
    priority: int
    target: Dict[str, Any]
    created_at: float = field(default_factory=lambda: time.time())
    status: str = 'pending'
    retries: int = 0


@dataclass
class RecoveryAction:
    action: str
    reason: str
    retryable: bool = True


@dataclass
class ObjectRecord:
    object_id: str
    label: str
    x: float
    y: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutiveState:
    mode: str = 'idle'
    stability: float = 1.0
    coherence: float = 1.0
    phase: float = 0.0
    domain: str = 'default'
    notes: Optional[str] = None