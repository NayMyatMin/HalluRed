import argparse
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from fastchat.model import get_conversation_template

from common_utils import (
    load_model_and_tokenizer,
    get_model_vals,
    compute_scores,
    def_dict_value,
    get_roc_auc_scores,
)


"""
SelfCheck evaluation removed for now per request. Keeping only FAVA-annotated evaluation.
"""


def evaluate_fava_annot(model, tokenizer, n_samples: int, use_toklens: bool, model_name_or_path: str):
    import utils_fava_annotated as fava

    df, _ = fava.get_fava_data(n_samples=n_samples or 200)

    system_prompt = ""
    chat_template = get_conversation_template(model_name_or_path)

    mt_list = ["logit", "attns"]
    scores = []
    indiv_scores = {mt: defaultdict(def_dict_value) for mt in mt_list}
    labels: List[int] = []

    for i in range(len(df)):
        row = df.loc[i]
        prompt = row["prompt"]
        response = row["output"]
        labels.append(1 if row["hallucinated"] == 1 else 0)

        chat_template.set_system_message(system_prompt.strip())
        chat_template.messages = []
        chat_template.append_message(chat_template.roles[0], prompt.strip())
        chat_template.append_message(chat_template.roles[1], response.strip())

        full_prompt = chat_template.get_prompt()
        user_prompt = full_prompt.split(response.strip())[0].strip()

        tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).input_ids

        logit, hidden_act, attn = get_model_vals(model, tok_in.to(model.device if hasattr(model, "device") else 0))
        logit = logit[0].cpu()
        hidden_act = [x[0].to(torch.float32).cpu() for x in hidden_act]
        attn = [x[0].to(torch.float32).cpu() for x in attn]
        tok_in = tok_in.cpu()

        tok_len = [tok_in_u.shape[1], tok_in.shape[1]]
        compute_scores(
            [logit],
            [hidden_act],
            [attn],
            scores,
            indiv_scores,
            mt_list,
            [tok_in],
            [tok_len],
            use_toklens=use_toklens,
        )

    y_true = np.array(labels).astype(int)
    summary = {}
    if "logit" in indiv_scores:
        for k in ["perplexity", "window_entropy", "logit_entropy"]:
            if k in indiv_scores["logit"]:
                auc, acc, tpr5, *_ = get_roc_auc_scores(np.array(indiv_scores["logit"][k]), y_true)
                summary[f"logit/{k}"] = (auc, acc, tpr5)
    if "attns" in indiv_scores:
        layer_keys = sorted([kk for kk in indiv_scores["attns"] if kk.startswith("Attn")], key=lambda x: int(x[4:]))
        if layer_keys:
            arr = np.stack([np.array(indiv_scores["attns"][kk]) for kk in layer_keys])
            s = arr.mean(0)
            auc, acc, tpr5, *_ = get_roc_auc_scores(s, y_true)
            summary["attns/mean_layers"] = (auc, acc, tpr5)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fava_annot"], required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--adapter_dir", type=str, default="", help="Path to LoRA adapter directory; leave empty for baseline")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--use_toklens", action="store_true")
    args = parser.parse_args()

    # Load model and tokenizer
    base_name = args.model
    model, tokenizer = load_model_and_tokenizer(base_name, dtype=torch.bfloat16)
    if args.adapter_dir:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_dir)
        except Exception as e:
            print("[WARN] Failed to load adapter; proceeding with base model. Error:", e)
    model.eval()

    summary = evaluate_fava_annot(model, tokenizer, args.n_samples, args.use_toklens, base_name)

    print("=== LLM-Check summary ===")
    for k, (auc, acc, tpr5) in summary.items():
        print(f"{k:20s}  AUC={auc:.3f}  ACC={acc:.3f}  TPR@5%FPR={tpr5:.3f}")


if __name__ == "__main__":
    main()


