from typing import Callable

import torch


def fgsm_embeddings(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pressure_fn: Callable[[torch.nn.modules.container.ModuleDict], torch.Tensor],
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute FGSM-step in embedding space using a pressure function over telemetry.

    The pressure_fn should perform a clean forward with inputs_embeds.requires_grad_(True),
    compute telemetry, aggregate to a scalar pressure, and return it for backprop.

    Returns:
        inputs_embeds_adv: perturbed embeddings tensor with same shape as input embeddings.
    """
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = embedding_layer(input_ids)

    embeds = base_embeds.clone().detach().requires_grad_(True)
    # Build kwargs for model forward
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    # The pressure function should aggregate outputs into a scalar pressure
    pressure = pressure_fn(outputs)
    grad = torch.autograd.grad(pressure, embeds, retain_graph=False, create_graph=False)[0]
    if grad is None:
        return base_embeds.detach()
    return (embeds + epsilon * torch.sign(grad)).detach()


