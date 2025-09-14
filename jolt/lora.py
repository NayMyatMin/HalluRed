from typing import Iterable, List


def attach_lora(model, target_modules: Iterable[str], r: int, alpha: int, dropout: float = 0.0):
    """Attach LoRA adapters to target linear modules using PEFT if available.

    Args:
        model: HF Transformers model
        target_modules: iterable of module name fragments to match (e.g., ["q_proj", "k_proj", ...])
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout

    Returns:
        Potentially wrapped model with LoRA adapters.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        print("[WARN] peft not installed; proceeding without LoRA")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        inference_mode=False,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


