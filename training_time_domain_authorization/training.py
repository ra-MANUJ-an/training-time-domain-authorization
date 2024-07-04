from typing import Any, List, Tuple

from loguru import logger
import torch
import numpy as np
from transformers import AutoModelForCausalLM, get_scheduler
from tqdm import tqdm

from training_time_domain_authorization.arguments import ARGS
from training_time_domain_authorization.datasets.datasets import BenchmarkDataset
from training_time_domain_authorization.device import accelerator
from training_time_domain_authorization.losses.causal_language_modeling_loss import (
    causal_language_modeling_loss,
)

np.object = object  # stability across NP versions

args = ARGS


def train_model(
    model_name: str,
    dataset: BenchmarkDataset,
) -> Tuple[Any, List[float]]:
    dtype = "auto"
    if "meta-llama" in model_name or "qwen" in model_name.lower():
        dtype = torch.bfloat16
    logger.info(f"Training model {model_name} on dataset {dataset.name}")
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(dataset.train_dataloader) * args.num_epochs

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.10,
        num_training_steps=num_training_steps,
    )
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    dataset.train_dataloader = accelerator.prepare(dataset.train_dataloader)
    dataset.test_dataloader = accelerator.prepare(dataset.test_dataloader)

    logger.info("Beginning training loop")
    # simple training loop
    logger.info("Evaluating model before training")
    results = dataset.evaluate(model)
    eval_datas = []
    eval_datas.append({**results, "step": 0, "epoch": 0})
    progress_bar = tqdm(
        range(args.num_epochs * len(dataset.train_dataloader)), desc="Training"
    )
    for epoch in range(args.num_epochs):
        step = 0
        for batch in dataset.train_dataloader:
            progress_bar.update(1)
            step += 1
            model.train()
            with accelerator.accumulate(model):
                loss = causal_language_modeling_loss(model, batch)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step != 0 and step % args.steps_to_eval == 0:
                results = dataset.evaluate(model)
                eval_datas.append({**results, "step": step, "epoch": epoch})
    # perform final evaluation
    results = dataset.evaluate(model)
    eval_datas.append({**results, "step": step, "epoch": epoch})
    return model, eval_datas
