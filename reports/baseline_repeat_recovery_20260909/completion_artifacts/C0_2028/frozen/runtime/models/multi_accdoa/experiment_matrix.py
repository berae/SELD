"""Single source of truth for the paper experiment matrix."""

DEFAULT_SEEDS = (2026, 2027, 2028)

VARIANTS = {
    "A0": {"causal": False, "use_velocity": False, "use_jepa": False},
    "A1": {"causal": False, "use_velocity": True, "use_jepa": False},
    "A2": {"causal": False, "use_velocity": False, "use_jepa": True},
    "A3": {"causal": False, "use_velocity": True, "use_jepa": True},
    "C0": {"causal": True, "use_velocity": False, "use_jepa": False},
    "C1": {"causal": True, "use_velocity": True, "use_jepa": False},
    "C2": {"causal": True, "use_velocity": False, "use_jepa": True},
    "C3": {"causal": True, "use_velocity": True, "use_jepa": True},
}

TASK_IDS = {
    "A0": {"full": "32", "smoke": "33"},
    "C0": {"full": "35", "smoke": "34"},
    "C1": {"full": "40", "smoke": "43"},
    "C2": {"full": "41", "smoke": "44"},
    "C3": {"full": "42", "smoke": "45"},
    "A1": {"full": "50", "smoke": "53"},
    "A2": {"full": "51", "smoke": "54"},
    "A3": {"full": "52", "smoke": "55"},
}

TASK_CONFIGS = {
    task_id: {
        "variant": variant,
        "stage": stage,
        **VARIANTS[variant],
    }
    for variant, stages in TASK_IDS.items()
    for stage, task_id in stages.items()
}


def get_task_config(task_id):
    """Return a copy so callers cannot mutate the global experiment matrix."""
    return dict(TASK_CONFIGS[str(task_id)])


def get_task_id(variant, stage="full"):
    return TASK_IDS[variant.upper()][stage]
