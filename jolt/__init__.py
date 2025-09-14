"""JOLT: Clean-data, detector-agnostic telemetry suppression.

This package provides:
- Telemetry metrics (NII, VEI) over hidden states and attentions
- Window selection utilities
- Adversarial embedding probes (FGSM)
- Telemetry matching loss
- LoRA attachment helpers
- Training orchestration and evaluation stubs
"""

__all__ = [
    "telemetry",
    "windows",
    "adv",
    "loss",
    "lora",
    "train",
]


