from typing import List, Tuple

import torch


def select_early_window(token_length: int, prompt_len: int, early_len: int) -> List[int]:
    """Return indices for an early window after the prompt boundary.

    Args:
        token_length: total sequence length (including prompt + answer)
        prompt_len: number of tokens belonging to the user prompt
        early_len: desired early window size

    Returns:
        List of token indices [start, ..., end) clamped to the sequence.
    """
    start = max(0, prompt_len)
    end = min(token_length, start + max(1, early_len))
    return list(range(start, end))


def select_answer_span(prompt_len: int, full_len: int) -> List[int]:
    """Approximate answer span as [prompt_len, full_len).

    For datasets without explicit span tags, we treat the assistant output
    as all tokens after the prompt boundary.
    """
    start = max(0, prompt_len)
    end = max(start + 1, full_len)
    return list(range(start, end))


def build_windows_from_toklens(tok_len_pair: Tuple[int, int], early_len: int) -> List[List[int]]:
    """Build both early and answer windows from (prompt_len, full_len).

    Args:
        tok_len_pair: (prompt_len, full_len) token counts
        early_len: early window size

    Returns:
        List of windows [early_window_indices, answer_window_indices]
    """
    prompt_len, full_len = tok_len_pair
    early = select_early_window(full_len, prompt_len, early_len)
    ans = select_answer_span(prompt_len, full_len)
    return [early, ans]


def indices_to_mask(indices: List[int], length: int, device: torch.device) -> torch.Tensor:
    """Create a boolean mask tensor of shape (length,) with True at given indices."""
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if len(indices) > 0:
        clamped = [i for i in indices if 0 <= i < length]
        if len(clamped) > 0:
            mask[torch.tensor(clamped, dtype=torch.long, device=device)] = True
    return mask


