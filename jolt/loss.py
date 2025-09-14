from typing import Dict, List, Union, Optional

import torch


def telemetry_loss(
    t_adv: Dict[str, Union[List[float], torch.Tensor]],
    t_clean: Dict[str, Union[List[float], torch.Tensor]],
    lambda_: float,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute L2 loss between adv and clean telemetry tensors (sum over keys).

    Accepts dicts with matching keys (e.g., 'nii', 'vei_hid', 'vei_att'). Each value can be a
    torch.Tensor (preferred) or a List[float]. Tensors are expected to be on the active device.
    """
    loss = torch.zeros((), dtype=torch.float32)
    for k in t_clean.keys():
        v1 = t_clean[k]
        v2 = t_adv[k]
        if not isinstance(v1, torch.Tensor):
            v1 = torch.tensor(v1, dtype=torch.float32)
        if not isinstance(v2, torch.Tensor):
            v2 = torch.tensor(v2, dtype=torch.float32)
        # Align lengths if needed
        if v1.numel() != v2.numel():
            n = max(v1.numel(), v2.numel())
            v1 = torch.nn.functional.pad(v1, (0, n - v1.numel()))
            v2 = torch.nn.functional.pad(v2, (0, n - v2.numel()))
        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        loss = loss + w * torch.mean((v2 - v1.detach()) ** 2)
    return lambda_ * loss


