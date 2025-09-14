from dataclasses import asdict
from typing import Dict, List, Tuple
import os
import re
import csv

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import JoltCfg
from .lora import attach_lora
from .telemetry import compute_nii, compute_vei_attn, compute_vei_hidden, compute_telemetry_torch
from .windows import select_answer_span
from .plot_telemetry import generate_plots
from .adv import fgsm_embeddings
from .loss import telemetry_loss


def _extract_qa_pairs(batch: Dict[str, str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if "question" in batch and "answers" in batch:
        for q, a in zip(batch["question"], batch["answers"]):
            if isinstance(a, dict) and "text" in a and len(a["text"]) > 0:
                ans = a["text"][0]
            elif isinstance(a, list) and len(a) > 0:
                ans = a[0]
            else:
                ans = str(a)
            pairs.append((q, ans))
    elif "question" in batch and "answer" in batch:
        for q, a in zip(batch["question"], batch["answer"]):
            pairs.append((q, str(a)))
    return pairs


def _collate_tokenize(tokenizer, texts: List[str], max_length: int = 1024):
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tok["labels"] = tok["input_ids"].clone()
    return tok


def run_training(cfg: JoltCfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[JOLT] Config:", asdict(cfg))

    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    model = attach_lora(model, cfg.lora.targets, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout)
    model.to(device)
    model.train()

    # Data
    ds = datasets.load_dataset(cfg.data.corpus, split=cfg.data.split)
    if cfg.data.limit:
        ds = ds.select(range(min(cfg.data.limit, len(ds))))

    def _map_to_text(batch):
        # Expect question/answers fields in SQuAD-like datasets
        pairs = _extract_qa_pairs(batch)
        user_texts = [f"question: {q}\nanswer:" for (q, _a) in pairs]
        full_texts = [f"question: {q}\nanswer: {_a}" for (q, _a) in pairs]
        return {"user_text": user_texts, "full_text": full_texts}

    text_ds = ds.map(_map_to_text, batched=True, remove_columns=ds.column_names)
    loader = DataLoader(text_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # CSV logging for plotting
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "jolt_telemetry.csv")
    # Overwrite CSV at the start of each run (avoid appending across runs)
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass

    step = 0
    for batch in loader:
        if step >= cfg.train.max_steps:
            break

        full_texts = batch["full_text"]
        user_texts = batch["user_text"]
        tok = _collate_tokenize(tokenizer, full_texts)
        tok = {k: v.to(device) for k, v in tok.items()}
        # Tokenize user prompts to get exact boundary
        user_tok = tokenizer(
            user_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tok["input_ids"].shape[1],
        )
        user_tok = {k: v.to(device) for k, v in user_tok.items()}

        # Clean forward with telemetry
        outputs = model(
            input_ids=tok["input_ids"],
            attention_mask=tok.get("attention_mask"),
            labels=tok["labels"],
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        loss_ce = outputs.loss

        # Build answer-only window using simple boundary heuristic
        # Here we do not have separate prompt/answer tokenization; approximate
        # Build per-sample answer windows (supports batch_size >= 1)
        B = tok["input_ids"].shape[0]
        windows_per_sample: List[List[int]] = []
        prompt_lens = []
        full_lens = []
        answer_lens = []
        for b in range(B):
            if "attention_mask" in tok:
                full_len_b = int(tok["attention_mask"][b].sum().item())
            else:
                full_len_b = int(tok["input_ids"].shape[1])
            if "attention_mask" in user_tok:
                prompt_len_b = int(user_tok["attention_mask"][b].sum().item())
            else:
                prompt_len_b = int(user_tok["input_ids"].shape[1])
            prompt_len_b = max(0, min(prompt_len_b, full_len_b - 1))
            aw = select_answer_span(prompt_len_b, full_len_b)
            windows_per_sample.append(aw)
            prompt_lens.append(prompt_len_b)
            full_lens.append(full_len_b)
            answer_lens.append(len(aw))
        # Aggregate simple batch means for logging
        prompt_len = int(sum(prompt_lens) / max(1, len(prompt_lens)))
        full_len = int(sum(full_lens) / max(1, len(full_lens)))
        answer_len = int(sum(answer_lens) / max(1, len(answer_lens)))

        # Differentiable telemetry for clean pass (detach for target)
        # Compute clean telemetry (average across batch when B>1)
        def _telemetry_mean(hidden_states, attentions, win_per_sample):
            acc = {"nii": [], "vei_hid": [], "vei_att": [], "hid_logvol": []}
            for b in range(hidden_states[1].shape[0]):
                hs_b = tuple(h[b:b+1] for h in hidden_states)
                at_b = tuple(a[b:b+1] for a in attentions)
                t_b = compute_telemetry_torch(
                    hs_b,
                    at_b,
                    [win_per_sample[b]],
                    weight_by_dx_magnitude=(cfg.telemetry.nii_weighting == "magnitude"),
                    min_dx_norm=cfg.telemetry.nii_min_norm,
                    head_sample=cfg.telemetry.head_sample,
                )
                for k in acc:
                    acc[k].append(t_b[k])
            return {k: torch.stack(v, dim=0).mean(dim=0) for k, v in acc.items()}

        t_clean_t = _telemetry_mean(outputs.hidden_states, outputs.attentions, windows_per_sample)
        t_clean = {
            "nii_win0": t_clean_t["nii"].detach().tolist(),
            "vei_att_win0": t_clean_t["vei_att"].detach().tolist(),
            "hid_logvol_win0": t_clean_t["hid_logvol"].detach().tolist(),
        }

        # FGSM: compute adversarial embeddings using pressure = sum per-layer telemetry
        # FGSM via helper utility: define pressure over model outputs
        def _pressure_fn(outputs):
            t_press = _telemetry_mean(outputs.hidden_states, outputs.attentions, windows_per_sample)
            w = getattr(cfg.loss, "weights", {}) or {}
            w_nii = float(w.get("nii", 1.0))
            w_va = float(w.get("vei_att", 1.0))
            w_hv = float(w.get("hid_logvol", 1.0))
            return (w_nii * t_press["nii"].sum() + w_va * t_press["vei_att"].sum() + w_hv * t_press["hid_logvol"].sum())

        adv_embeds = fgsm_embeddings(
            model,
            tok["input_ids"],
            tok.get("attention_mask"),
            _pressure_fn,
            epsilon=cfg.adv.epsilon,
        )

        # Adversarial forward and telemetry
        outputs_adv = model(
            inputs_embeds=adv_embeds,
            attention_mask=tok.get("attention_mask"),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        t_adv_t = _telemetry_mean(outputs_adv.hidden_states, outputs_adv.attentions, windows_per_sample)
        # Telemetry matching loss (match adv to clean target); drop vei_hid; add hid_logvol
        weights = getattr(cfg.loss, "weights", {}) or {}
        tm_loss = telemetry_loss(
            {"nii": t_adv_t["nii"], "vei_att": t_adv_t["vei_att"], "hid_logvol": t_adv_t["hid_logvol"]},
            {"nii": t_clean_t["nii"], "vei_att": t_clean_t["vei_att"], "hid_logvol": t_clean_t["hid_logvol"]},
            lambda_=cfg.loss.lambda_,
            weights=weights,
        )

        loss = loss_ce + tm_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % cfg.train.eval_every == 0:
            # Layerwise telemetry logging + summary stats for the answer window (win0)
            def _get_layer_vals(d: Dict[str, List[float]], key: str) -> List[float]:
                vals = d.get(key, [])
                if not isinstance(vals, list):
                    return []
                return [float(x) for x in vals if x == x]

            def _fmt_list(vals: List[float]) -> str:
                return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"

            def _summ(vals: List[float]) -> str:
                if len(vals) == 0:
                    return "mean=nan,min=nan,max=nan"
                m = float(sum(vals) / len(vals))
                return f"mean={m:.3f},min={min(vals):.3f},max={max(vals):.3f}"

            nii_layers = _get_layer_vals(t_clean, "nii_win0")
            hv_layers = _get_layer_vals(t_clean, "hid_logvol_win0")
            va_layers = _get_layer_vals(t_clean, "vei_att_win0")

            print(f"[JOLT] step {step} loss_ce={loss_ce.item():.4f}")
            print(f"  nii(ans) layers: {_fmt_list(nii_layers)} | {_summ(nii_layers)}")
            print(f"  hid_logvol(ans) layers: {_fmt_list(hv_layers)} | {_summ(hv_layers)}")
            print(f"  vei_att(ans) layers: {_fmt_list(va_layers)} | {_summ(va_layers)}")

            # Append to CSV for plotting
            header = [
                "step",
                "loss_ce",
                "tm_loss",
                "prompt_len",
                "full_len",
                "answer_len",
                "nii_layers",
                "vei_att_layers",
                "hid_logvol_layers",
                "adv_nii_layers",
                "adv_vei_att_layers",
                "adv_hid_logvol_layers",
                "delta_nii_layers",
                "delta_vei_att_layers",
                "delta_hid_logvol_layers",
            ]
            adv_nii_layers = t_adv_t["nii"].detach().cpu().tolist()
            adv_va_layers = t_adv_t["vei_att"].detach().cpu().tolist()
            adv_hv_layers = t_adv_t["hid_logvol"].detach().cpu().tolist()
            # deltas (adv - clean)
            d_nii = (t_adv_t["nii"] - t_clean_t["nii"].detach()).detach().cpu().tolist()
            d_va = (t_adv_t["vei_att"] - t_clean_t["vei_att"].detach()).detach().cpu().tolist()
            d_hv = (t_adv_t["hid_logvol"] - t_clean_t["hid_logvol"].detach()).detach().cpu().tolist()

            row = [
                step,
                float(loss_ce.item()),
                float(tm_loss.item()),
                int(prompt_len),
                int(full_len),
                int(answer_len),
                " ".join(f"{v:.6f}" for v in nii_layers),
                " ".join(f"{v:.6f}" for v in va_layers),
                " ".join(f"{v:.6f}" for v in t_clean_t["hid_logvol"].detach().cpu().tolist()),
                " ".join(f"{v:.6f}" for v in adv_nii_layers),
                " ".join(f"{v:.6f}" for v in adv_va_layers),
                " ".join(f"{v:.6f}" for v in adv_hv_layers),
                " ".join(f"{v:.6f}" for v in d_nii),
                " ".join(f"{v:.6f}" for v in d_va),
                " ".join(f"{v:.6f}" for v in d_hv),
            ]
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow(row)
        step += 1

    print("[JOLT] Training completed up to max_steps.")
    # Save LoRA checkpoint under a directory that includes the model name
    try:
        def _slugify_model_name(name: str) -> str:
            s = name.strip().replace("/", "_").replace(":", "_").replace(" ", "_")
            s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
            s = re.sub(r"_+", "_", s).strip("_")
            return s or "model"

        base_dir = cfg.train.save_dir
        model_slug = _slugify_model_name(cfg.model)
        final_dir = base_dir if os.path.basename(os.path.normpath(base_dir)) == model_slug else os.path.join(base_dir, model_slug)
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        print(f"[JOLT] LoRA checkpoint saved to {final_dir}")
    except Exception as e:
        print("[WARN] Failed to save LoRA checkpoint:", e)
    # Auto-generate plots if CSV exists
    if os.path.exists(csv_path):
        try:
            generate_plots(csv_path, os.path.dirname(csv_path))
            print(f"[JOLT] Plots saved to {os.path.dirname(csv_path)}")
        except Exception as e:
            print("[WARN] Plot generation failed:", e)


