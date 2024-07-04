def causal_language_modeling_loss(model, batch):
    outputs = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["input_ids"],
    )
    loss = outputs.loss
    return loss
