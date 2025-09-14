from typing import Dict, List

import torch


def telemetry_loss(t_adv: Dict[str, List[float]], t_clean: Dict[str, List[float]], device: torch.device, lambda_: float) -> torch.Tensor:
    """Compute L2 loss between adv and clean telemetry dictionaries.

    Expects both dicts to have identical keys (e.g., win0, win1) and same layer lengths.
    """
    loss = torch.zeros((), device=device)
    for w in t_clean.keys():
        v1 = torch.tensor(t_clean[w], dtype=torch.float32, device=device)
        v2 = torch.tensor(t_adv[w], dtype=torch.float32, device=device)
        # Pad to same length just in case
        if v1.numel() != v2.numel():
            n = max(v1.numel(), v2.numel())
            v1 = torch.nn.functional.pad(v1, (0, n - v1.numel()))
            v2 = torch.nn.functional.pad(v2, (0, n - v2.numel()))
        loss = loss + torch.mean((v1 - v2) ** 2)
    return lambda_ * loss


