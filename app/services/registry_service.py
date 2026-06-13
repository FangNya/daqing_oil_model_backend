import json
from copy import deepcopy

from app.config import REGISTRY_PATH, TARGET_SPECS, VALID_CYCLES, get_rolling_cycle


def load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def get_target_config(target):
    if target not in TARGET_SPECS:
        raise ValueError(f"INVALID_TARGET: {target}")
    registry = load_registry()
    merged = deepcopy(TARGET_SPECS[target])
    merged.update(registry.get(target, {}))
    merged["default_cycle"] = get_rolling_cycle(target, merged["default_cycle"])
    return merged


def resolve_cycle(target, requested_cycle):
    cfg = get_target_config(target)
    cycle = cfg["default_cycle"]
    if cycle not in VALID_CYCLES:
        raise ValueError(f"INVALID_CYCLE: {cycle}")
    return cycle
