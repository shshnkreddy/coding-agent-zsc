"""Explicit inner-search guidance configurations, independent of SP/XP."""

GUIDANCE_MODES = (
    "standalone_no_guidance",
    "meta_baseline",
    "meta_guided",
    "standalone_guided",
)


def validate_guidance_mode(mode, guidance, meta_update_id=None):
    if mode not in GUIDANCE_MODES:
        raise ValueError(f"unknown guidance mode: {mode}")
    guided = mode in {"meta_guided", "standalone_guided"}
    if bool(guidance.strip()) != guided:
        raise ValueError(f"{mode} requires {'nonempty' if guided else 'empty'} guidance")
    if mode.startswith("standalone") and meta_update_id:
        raise ValueError("standalone guidance modes cannot have a meta update ID")
    return guided
