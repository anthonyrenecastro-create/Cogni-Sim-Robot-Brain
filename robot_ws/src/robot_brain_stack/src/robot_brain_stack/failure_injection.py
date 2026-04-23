import random


def should_inject_failure(enabled: bool, rate: float) -> bool:
    if not enabled:
        return False
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return random.random() < rate
