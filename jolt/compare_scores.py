import argparse
import os
import pickle as pk
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from common_utils import get_roc_auc_scores


def _metrics_from_pickle(path: str, balanced: bool) -> Tuple[Dict[str, Tuple[float, float, float]], np.ndarray, Dict[str, List[float]]]:
    with open(path, "rb") as f:
        _scores, sample_indiv_scores, sample_labels = pk.load(f)

    labels = np.array(sample_labels).astype(int)
    if balanced:
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        idx = np.concatenate([pos_idx[:n], neg_idx[:n]])
        idx.sort()
    else:
        idx = np.arange(len(labels))

    out: Dict[str, Tuple[float, float, float]] = {}

    # Logit
    if "logit" in sample_indiv_scores:
        for k in ["perplexity", "window_entropy", "logit_entropy"]:
            if k in sample_indiv_scores["logit"]:
                vals = np.array(sample_indiv_scores["logit"][k])[idx]
                if k == "perplexity":
                    vals = -vals
                auc, acc, tpr5, *_ = get_roc_auc_scores(vals, labels[idx])
                out[f"logit/{k}"] = (auc, acc, tpr5)

    # Attns
    if "attns" in sample_indiv_scores:
        layer_keys = sorted(
            [kk for kk in sample_indiv_scores["attns"] if kk.startswith("Attn")],
            key=lambda x: int(x[4:]),
        )
        if layer_keys:
            arr = np.stack([np.array(sample_indiv_scores["attns"][kk])[idx] for kk in layer_keys])
            arr = -arr
            s = arr.mean(0)
            auc, acc, tpr5, *_ = get_roc_auc_scores(s, labels[idx])
            out["attns/mean_layers"] = (auc, acc, tpr5)
            for kk in layer_keys:
                v = -np.array(sample_indiv_scores["attns"][kk])[idx]
                auc, acc, tpr5, *_ = get_roc_auc_scores(v, labels[idx])
                out[f"attns/{kk}"] = (auc, acc, tpr5)

    # Hidden
    if "hidden" in sample_indiv_scores:
        hkeys = sorted(
            [kk for kk in sample_indiv_scores["hidden"] if kk.startswith("Hly")],
            key=lambda x: int(x[3:]),
        )
        if hkeys:
            arr = np.stack([np.array(sample_indiv_scores["hidden"][kk])[idx] for kk in hkeys])
            arr = -arr
            s = arr.mean(0)
            auc, acc, tpr5, *_ = get_roc_auc_scores(s, labels[idx])
            out["hidden/mean_layers"] = (auc, acc, tpr5)
            for kk in hkeys:
                v = -np.array(sample_indiv_scores["hidden"][kk])[idx]
                auc, acc, tpr5, *_ = get_roc_auc_scores(v, labels[idx])
                out[f"hidden/{kk}"] = (auc, acc, tpr5)

    return out, labels[idx], sample_indiv_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_pkl", type=str, required=True)
    p.add_argument("--adapted_pkl", type=str, required=True)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--print_layerwise", action="store_true")
    p.add_argument("--plot_dir", type=str, default="", help="If set, write delta plots to this directory")
    p.add_argument("--metric", type=str, default="auc", choices=["auc", "acc", "tpr"], help="Metric to plot for deltas")
    args = p.parse_args()

    base, _, _ = _metrics_from_pickle(args.baseline_pkl, args.balanced)
    adap, _, _ = _metrics_from_pickle(args.adapted_pkl, args.balanced)

    # Only show layerwise if requested
    def _visible(k: str) -> bool:
        if args.print_layerwise:
            return True
        return ("/mean_layers" in k) or k.startswith("logit/")

    keys = sorted(set(base.keys()) | set(adap.keys()), key=lambda x: (x.split("/")[0], x))

    print("=== Baseline vs Adapted (delta = adapted - baseline) ===")
    for k in keys:
        if not _visible(k):
            continue
        b = base.get(k)
        a = adap.get(k)
        if b is None or a is None:
            continue
        da = a[0] - b[0]
        dacc = a[1] - b[1]
        dtpr = a[2] - b[2]
        print(f"{k:24s}  base(AUC={b[0]:.3f},ACC={b[1]:.3f},TPR5={b[2]:.3f})  "
              f"adapt(AUC={a[0]:.3f},ACC={a[1]:.3f},TPR5={a[2]:.3f})  "
              f"delta(AUC={da:+.3f},ACC={dacc:+.3f},TPR5={dtpr:+.3f})")

    # Optional plotting
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        def pick_idx(t):
            return {"auc": 0, "acc": 1, "tpr": 2}[t]
        midx = pick_idx(args.metric)

        # Summary bar (logit + mean layers)
        summary_keys = [
            "logit/perplexity",
            "logit/window_entropy",
            "logit/logit_entropy",
            "attns/mean_layers",
            "hidden/mean_layers",
        ]
        labels = []
        deltas = []
        for k in summary_keys:
            if k in base and k in adap:
                labels.append(k)
                deltas.append(adap[k][midx] - base[k][midx])
        if labels:
            plt.clf()
            x = np.arange(len(labels))
            plt.bar(x, deltas)
            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylabel(f"Delta {args.metric.upper()}")
            plt.title("Adapted - Baseline (summary)")
            plt.grid(axis="y", linestyle=":")
            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f"compare_summary_delta_{args.metric}.png"), dpi=200)

        # Attn per-layer delta line
        attn_layers = [k for k in keys if k.startswith("attns/Attn")]
        if attn_layers:
            attn_layers_sorted = sorted(attn_layers, key=lambda x: int(x.split("Attn")[-1]))
            y = [adap[k][midx] - base[k][midx] for k in attn_layers_sorted]
            plt.clf()
            plt.plot(range(1, len(y) + 1), y, marker="o")
            plt.xlabel("Layer")
            plt.ylabel(f"Delta {args.metric.upper()}")
            plt.title("Attn layer delta (adapted - baseline)")
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f"compare_attn_delta_{args.metric}.png"), dpi=200)

        # Hidden per-layer delta line
        hid_layers = [k for k in keys if k.startswith("hidden/Hly")]
        if hid_layers:
            hid_layers_sorted = sorted(hid_layers, key=lambda x: int(x.split("Hly")[-1]))
            y = [adap[k][midx] - base[k][midx] for k in hid_layers_sorted]
            plt.clf()
            plt.plot(range(1, len(y) + 1), y, marker="o")
            plt.xlabel("Layer")
            plt.ylabel(f"Delta {args.metric.upper()}")
            plt.title("Hidden layer delta (adapted - baseline)")
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f"compare_hidden_delta_{args.metric}.png"), dpi=200)


if __name__ == "__main__":
    main()


