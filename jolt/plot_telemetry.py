from typing import List, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_series(col: pd.Series) -> Tuple[np.ndarray, int]:
    lists: List[List[float]] = []
    max_len = 0
    for s in col.fillna(""):
        vals = [float(x) for x in str(s).strip().split() if x != ""]
        lists.append(vals)
        max_len = max(max_len, len(vals))
    arr = np.full((len(lists), max_len), np.nan, dtype=float)
    for i, vals in enumerate(lists):
        if len(vals) > 0:
            arr[i, : len(vals)] = vals
    return arr, max_len


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_plots(csv_path: str, out_dir: str):
    _ensure_dir(out_dir)
    df = pd.read_csv(csv_path)
    steps_all = df["step"].to_numpy()
    # If the CSV contains multiple runs appended, select the last contiguous run block
    # determined by where step resets (diff < 0).
    if len(steps_all) > 1:
        diffs = np.diff(steps_all)
        resets = np.where(diffs < 0)[0]
        start_idx = int(resets[-1] + 1) if resets.size > 0 else 0
    else:
        start_idx = 0
    df = df.iloc[start_idx:].reset_index(drop=True)
    steps = df["step"].to_numpy()

    nii, _ = _parse_series(df["nii_layers"]) if "nii_layers" in df.columns else (np.zeros((0, 0)), 0)
    va, _ = _parse_series(df["vei_att_layers"]) if "vei_att_layers" in df.columns else (np.zeros((0, 0)), 0)
    hv, _ = _parse_series(df["hid_logvol_layers"]) if "hid_logvol_layers" in df.columns else (np.zeros((0, 0)), 0)

    has_adv = all(c in df.columns for c in [
        "adv_nii_layers", "adv_vei_att_layers", "adv_hid_logvol_layers"
    ])
    adv_nii, adv_va, adv_hv = (None, None, None)
    if has_adv:
        adv_nii, _ = _parse_series(df["adv_nii_layers"])  # shape: (steps, L)
        adv_va, _ = _parse_series(df["adv_vei_att_layers"]) 
        adv_hv, _ = _parse_series(df["adv_hid_logvol_layers"]) 

    has_delta = all(c in df.columns for c in [
        "delta_nii_layers", "delta_vei_att_layers", "delta_hid_logvol_layers"
    ])
    d_nii, d_va, d_hv = (None, None, None)
    if has_delta:
        d_nii, _ = _parse_series(df["delta_nii_layers"]) 
        d_va, _ = _parse_series(df["delta_vei_att_layers"]) 
        d_hv, _ = _parse_series(df["delta_hid_logvol_layers"]) 

    # Heatmaps
    def heatmap(arr: np.ndarray, title: str, fname: str):
        plt.figure(figsize=(10, 4))
        plt.imshow(arr.T, aspect="auto", origin="lower")
        plt.colorbar(label=title)
        plt.xlabel("step idx")
        plt.ylabel("layer")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    # Essential heatmaps: delta NII, delta VEI_att, delta hid_logvol (adv - clean)
    if has_delta and d_nii is not None and d_nii.size:
        heatmap(d_nii, "Delta NII (adv - clean)", "jolt_delta_nii_heatmap.png")
    if has_delta and d_va is not None and d_va.size:
        heatmap(d_va, "Delta VEI_att (adv - clean)", "jolt_delta_vei_att_heatmap.png")
    if has_delta and d_hv is not None and d_hv.size:
        heatmap(d_hv, "Delta hid_logvol (adv - clean)", "jolt_delta_hid_logvol_heatmap.png")

    # Layer-mean trends
    # Essential trend: clean vs adv layer-mean comparison

    if has_adv and ((nii.size and adv_nii is not None) or (va.size and adv_va is not None) or (hv.size and adv_hv is not None)):
        plt.figure(figsize=(8, 4))
        if nii.size and adv_nii is not None:
            plt.plot(steps, np.nanmean(nii, axis=1), label="nii(clean)")
            plt.plot(steps, np.nanmean(adv_nii, axis=1), label="nii(adv)")
        if va.size and adv_va is not None:
            plt.plot(steps, np.nanmean(va, axis=1), label="vei_att(clean)")
            plt.plot(steps, np.nanmean(adv_va, axis=1), label="vei_att(adv)")
        if hv.size and adv_hv is not None:
            plt.plot(steps, np.nanmean(hv, axis=1), label="hid_logvol(clean)")
            plt.plot(steps, np.nanmean(adv_hv, axis=1), label="hid_logvol(adv)")
        plt.xlabel("step")
        plt.ylabel("layer-mean value")
        plt.title("Telemetry trends (clean vs adv)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "jolt_layer_mean_clean_adv.png"), dpi=160)
        plt.close()

    # Loss curves if available
    if "loss_ce" in df.columns:
        plt.figure(figsize=(8, 3))
        plt.plot(steps, df["loss_ce"].to_numpy(), label="loss_ce")
        if "tm_loss" in df.columns:
            plt.plot(steps, df["tm_loss"].to_numpy(), label="tm_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "jolt_losses.png"), dpi=160)
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/jolt_telemetry.csv")
    parser.add_argument("--out", type=str, default="logs")
    args = parser.parse_args()
    generate_plots(args.csv, args.out)


