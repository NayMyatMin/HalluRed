import argparse
import pickle as pk
from typing import Dict, List

import numpy as np

from common_utils import get_roc_auc_scores


def _print_summary(tag: str, scores: np.ndarray, labels: np.ndarray) -> None:
    auc, acc, tpr5, *_ = get_roc_auc_scores(scores, labels)
    print(f"{tag:20s}  AUC={auc:.3f}  ACC={acc:.3f}  TPR@5%FPR={tpr5:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_pkl", type=str, required=True)
    p.add_argument("--balanced", action="store_true", help="Balance pos/neg before computing metrics")
    p.add_argument("--print_layerwise", action="store_true", help="Print per-layer metrics for attns/hidden")
    args = p.parse_args()

    with open(args.scores_pkl, "rb") as f:
        scores, sample_indiv_scores, sample_labels = pk.load(f)

    labels = np.array(sample_labels).astype(int)

    # Optional balance: equalize pos/neg counts
    if args.balanced:
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        idx = np.concatenate([pos_idx[:n], neg_idx[:n]])
        idx.sort()
    else:
        idx = np.arange(len(labels))

    print("=== LLM-Check summary (from scores pickle) ===")

    # Logit metrics
    if "logit" in sample_indiv_scores:
        for k in ["perplexity", "window_entropy", "logit_entropy"]:
            if k in sample_indiv_scores["logit"]:
                vals = np.array(sample_indiv_scores["logit"][k])[idx]
                # Notebook orientation: use negative for perplexity, positive for others
                if k == "perplexity":
                    vals = -vals
                _print_summary(f"logit/{k}", vals, labels[idx])

    # Attention metrics
    if "attns" in sample_indiv_scores:
        layer_keys = sorted(
            [kk for kk in sample_indiv_scores["attns"] if kk.startswith("Attn")],
            key=lambda x: int(x[4:]),
        )
        if layer_keys:
            arr = np.stack([np.array(sample_indiv_scores["attns"][kk])[idx] for kk in layer_keys])
            # Notebook orientation: negate attention scores
            arr = -arr
            s = arr.mean(0)
            _print_summary("attns/mean_layers", s, labels[idx])
            if args.print_layerwise:
                for kk in layer_keys:
                    vals = -np.array(sample_indiv_scores["attns"][kk])[idx]
                    _print_summary(f"attns/{kk}", vals, labels[idx])

    # Hidden metrics
    if "hidden" in sample_indiv_scores:
        hkeys = sorted(
            [kk for kk in sample_indiv_scores["hidden"] if kk.startswith("Hly")],
            key=lambda x: int(x[3:]),
        )
        if hkeys:
            arr = np.stack([np.array(sample_indiv_scores["hidden"][kk])[idx] for kk in hkeys])
            # Notebook orientation: negate hidden scores
            arr = -arr
            s = arr.mean(0)
            _print_summary("hidden/mean_layers", s, labels[idx])
            if args.print_layerwise:
                for kk in hkeys:
                    vals = -np.array(sample_indiv_scores["hidden"][kk])[idx]
                    _print_summary(f"hidden/{kk}", vals, labels[idx])


if __name__ == "__main__":
    main()


