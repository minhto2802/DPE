import torch
import transformers


def bert_adamw_optimizer(model, lr, momentum, weight_decay):
    # Adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    del momentum
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    # optimizer = transformers.AdamW(
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def bert_lr_scheduler(optimizer, num_steps):
    return transformers.get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_steps)