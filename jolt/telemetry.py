from typing import Dict, List, Tuple

import torch


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def compute_nii(
    hidden_states: Tuple[torch.Tensor, ...],
    attentions: Tuple[torch.Tensor, ...],
    windows: List[List[int]],
    weight_by_dx_magnitude: bool = False,
    min_dx_norm: float = 0.0,
    head_sample: int = 0,
) -> Dict[str, List[float]]:
    """Compute Novelty Injection Index per layer for given windows.

    Args:
        hidden_states: tuple length L+1, each (B, T, D); index 0 is embeddings, 1..L are layers
        attentions: tuple length L, each (B, H, T, T)
        windows: list of index lists, e.g., [early_indices, answer_indices]

    Returns:
        Dict mapping window name to list of per-layer scalars (layers 1..L).
    """
    assert len(hidden_states) >= 2, "Need embeddings + at least one layer"
    B, T, D = hidden_states[1].shape
    Hn = attentions[0].shape[1] if len(attentions) > 0 else 1

    results: Dict[str, List[float]] = {}
    window_names = [f"win{i}" for i in range(len(windows))]
    for wname in window_names:
        results[wname] = []

    with torch.no_grad():
        # Pre-aggregate attentions for speed (mean or first-k heads)
        head_mean = []
        for att in attentions:
            if head_sample and att.shape[1] > head_sample:
                head_mean.append(att[:, :head_sample].mean(dim=1))
            else:
                head_mean.append(att.mean(dim=1))
        # For each layer, compute ΔX and attention context C = A X
        for layer_idx in range(1, len(hidden_states)):
            X_prev = hidden_states[layer_idx - 1]  # (B,T,D)
            X_curr = hidden_states[layer_idx]
            dX = X_curr - X_prev  # (B,T,D)
            A = head_mean[layer_idx - 1] if layer_idx - 1 < len(head_mean) else None
            if A is None:
                # Fallback: zero novelty if attentions are missing
                for wname in window_names:
                    results[wname].append(0.0)
                continue
            C = torch.matmul(A, X_curr)  # (B,T,T) @ (B,T,D) -> (B,T,D)

            # Compute per-window mean of 1 - cos^2(dX, C)
            for wname, idxs in zip(window_names, windows):
                if len(idxs) == 0:
                    results[wname].append(0.0)
                    continue
                idx = torch.tensor(idxs, dtype=torch.long, device=X_curr.device)
                dx_block = dX[:, idx, :]
                dx_norm = torch.norm(dx_block, dim=-1)  # (B,m)
                # optional thresholding
                if min_dx_norm > 0:
                    mask = (dx_norm >= min_dx_norm).float()
                else:
                    mask = torch.ones_like(dx_norm)
                dx_sel = _safe_normalize(dx_block)  # (B,m,D)
                c_sel = _safe_normalize(C[:, idx, :])
                cos = (dx_sel * c_sel).sum(dim=-1)  # (B,m)
                gamma = 1.0 - cos.pow(2.0)  # (B,m)
                if weight_by_dx_magnitude:
                    # scale by normalized norm (avoid inflating by big norms; cap at 1)
                    scale = (dx_norm / (dx_norm.mean() + 1e-8)).clamp_max(1.0)
                    gamma = gamma * scale * mask
                else:
                    gamma = gamma * mask
                denom = mask.sum().clamp_min(1.0)
                val = (gamma.sum() / denom).item()
                results[wname].append(val)

    return results


def _participation_ratio(C: torch.Tensor, eps: float = 1e-5) -> float:
    # C is (m,m), symmetric PSD with jitter added upstream
    tr = torch.trace(C)
    tr2 = torch.sum(C * C)
    return (tr * tr / (tr2 + eps)).item()


def compute_vei_hidden(
    hidden_states: Tuple[torch.Tensor, ...],
    windows: List[List[int]],
    eps: float = 1e-5,
) -> Dict[str, List[float]]:
    """Compute hidden VEI per layer for given windows."""
    assert len(hidden_states) >= 2
    results: Dict[str, List[float]] = {f"win{i}": [] for i in range(len(windows))}
    with torch.no_grad():
        for layer_idx in range(1, len(hidden_states)):
            X = hidden_states[layer_idx]  # (B,T,D)
            for wname, idxs in results.items():
                pass  # structure ensured
            for i, idxs in enumerate(windows):
                if len(idxs) == 0:
                    results[f"win{i}"].append(0.0)
                    continue
                idx = torch.tensor(idxs, dtype=torch.long, device=X.device)
                Xm = X[:, idx, :]  # (B,m,D)
                # compute per-batch and average
                vals: List[float] = []
                for b in range(Xm.shape[0]):
                    Z = Xm[b].to(torch.float32)  # (m,D)
                    m = Z.shape[0]
                    C = (Z @ Z.t()) / max(1, m)
                    C = C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
                    vals.append(_participation_ratio(C, eps=eps))
                results[f"win{i}"].append(float(sum(vals) / max(1, len(vals))))
    return results


def compute_vei_attn(
    attentions: Tuple[torch.Tensor, ...],
    windows: List[List[int]],
    eps: float = 1e-5,
    head_sample: int = 0,
) -> Dict[str, List[float]]:
    """Compute attention VEI per layer for given windows."""
    assert len(attentions) >= 1
    results: Dict[str, List[float]] = {f"win{i}": [] for i in range(len(windows))}
    with torch.no_grad():
        for layer_idx in range(len(attentions)):
            A = attentions[layer_idx]  # (B,H,T,T)
            if head_sample and A.shape[1] > head_sample:
                Amean = A[:, :head_sample].mean(dim=1)
            else:
                Amean = A.mean(dim=1)  # (B,T,T)
            for i, idxs in enumerate(windows):
                if len(idxs) == 0:
                    results[f"win{i}"].append(0.0)
                    continue
                idx = torch.tensor(idxs, dtype=torch.long, device=Amean.device)
                vals: List[float] = []
                for b in range(Amean.shape[0]):
                    K = Amean[b][idx][:, idx]  # (m,m)
                    L = torch.tril(K)
                    Ksym = (L + L.t()) * 0.5
                    Ksym = Ksym + eps * torch.eye(Ksym.shape[0], device=Ksym.device, dtype=Ksym.dtype)
                    vals.append(_participation_ratio(Ksym, eps=eps))
                results[f"win{i}"].append(float(sum(vals) / max(1, len(vals))))
    return results


# ===== Differentiable telemetry for training (returns tensors, no no_grad) =====

def compute_telemetry_torch(
    hidden_states: Tuple[torch.Tensor, ...],
    attentions: Tuple[torch.Tensor, ...],
    windows: List[List[int]],
    eps: float = 1e-5,
    weight_by_dx_magnitude: bool = False,
    min_dx_norm: float = 0.0,
    head_sample: int = 0,
) -> Dict[str, torch.Tensor]:
    """Differentiable telemetry (per-layer) for NII, VEI_hidden, VEI_attn.

    Returns dict with keys: 'nii', 'vei_hid', 'vei_att' mapping to tensors of shape (L,),
    where L is the number of transformer layers.
    """
    assert len(hidden_states) >= 2
    L = len(hidden_states) - 1
    # Build index tensor once per window (batch_size assumed 1 for training loop)
    # We use only first window (answer-only) during training integration.
    if len(windows) == 0 or len(windows[0]) == 0:
        # Degenerate: return zeros
        device = hidden_states[1].device
        zero = torch.zeros(L, dtype=torch.float32, device=device)
        return {"nii": zero, "vei_hid": zero, "vei_att": zero}
    idx = torch.tensor(windows[0], dtype=torch.long, device=hidden_states[1].device)

    # Head-mean attentions (or first-k heads)
    head_mean = []
    for att in attentions:
        if head_sample and att.shape[1] > head_sample:
            head_mean.append(att[:, :head_sample].mean(dim=1))
        else:
            head_mean.append(att.mean(dim=1))

    nii_vals: List[torch.Tensor] = []
    vh_vals: List[torch.Tensor] = []
    va_vals: List[torch.Tensor] = []

    for layer_idx in range(1, len(hidden_states)):
        X_prev = hidden_states[layer_idx - 1]  # (B,T,D)
        X_curr = hidden_states[layer_idx]
        dX = X_curr - X_prev  # (B,T,D)
        A = head_mean[layer_idx - 1] if layer_idx - 1 < len(head_mean) else None
        if A is None:
            device = X_curr.device
            zero = torch.zeros((), dtype=torch.float32, device=device)
            nii_vals.append(zero)
            vh_vals.append(zero)
            va_vals.append(zero)
            continue

        # NII
        C = torch.matmul(A, X_curr)  # (B,T,D)
        dx_block = dX[:, idx, :]
        dx_norm = torch.norm(dx_block, dim=-1)  # (B,m)
        if min_dx_norm > 0:
            mask = (dx_norm >= min_dx_norm).float()
        else:
            mask = torch.ones_like(dx_norm)
        dx_sel = _safe_normalize(dx_block)
        c_sel = _safe_normalize(C[:, idx, :])
        cos = (dx_sel * c_sel).sum(dim=-1)  # (B,m)
        gamma = 1.0 - cos.pow(2.0)
        if weight_by_dx_magnitude:
            scale = (dx_norm / (dx_norm.mean() + 1e-8)).clamp_max(1.0)
            gamma = gamma * scale * mask
        else:
            gamma = gamma * mask
        denom = mask.sum().clamp_min(1.0)
        nii_layer = gamma.sum() / denom
        nii_vals.append(nii_layer)

        # VEI_hidden
        Xm = X_curr[:, idx, :]  # (B,m,D)
        # Average PR across batch
        vh_acc = []
        for b in range(Xm.shape[0]):
            Z = Xm[b].to(torch.float32)
            m = max(1, Z.shape[0])
            C_h = (Z @ Z.t()) / m
            C_h = C_h + eps * torch.eye(C_h.shape[0], device=Z.device, dtype=Z.dtype)
            tr = torch.trace(C_h)
            tr2 = torch.sum(C_h * C_h)
            vh_acc.append((tr * tr) / (tr2 + eps))
        vh_vals.append(torch.stack(vh_acc).mean())

        # VEI_attention
        Amean = A  # (B,T,T)
        va_acc = []
        for b in range(Amean.shape[0]):
            K = Amean[b][idx][:, idx]
            Ltri = torch.tril(K)
            Ksym = 0.5 * (Ltri + Ltri.t())
            Ksym = Ksym + eps * torch.eye(Ksym.shape[0], device=Ksym.device, dtype=Ksym.dtype)
            tr = torch.trace(Ksym)
            tr2 = torch.sum(Ksym * Ksym)
            va_acc.append((tr * tr) / (tr2 + eps))
        va_vals.append(torch.stack(va_acc).mean())

    return {
        "nii": torch.stack(nii_vals),
        "vei_hid": torch.stack(vh_vals),
        "vei_att": torch.stack(va_vals),
    }


