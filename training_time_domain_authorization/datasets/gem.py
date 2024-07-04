from typing import Literal

import gem_metrics
from datasets import load_dataset
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from training_time_domain_authorization.arguments import ARGS
from training_time_domain_authorization.datasets.datasets import BenchmarkDataset

DEFAULT_BATCH_SIZE = 12
CONTEXT_LENGTH = 256


def construct_gem_dataset(
    dataset: Literal["viggo", "xsum", "cochrane-simplification", "common_gen"],
    tokenizer,
    test_batch_size: int = DEFAULT_BATCH_SIZE,
    train_batch_size: int = DEFAULT_BATCH_SIZE,
):
    train_ds = load_dataset("GEM/" + dataset, split="train")
    test_ds = load_dataset("GEM/" + dataset, split="test")

    dataset_structure = {
        "viggo": {
            "target_ds_label": "target",
            "target_prompt_label": "Description:",
            "input_ds_label": "meaning_representation",
            "input_prompt_label": "Meaning Representation:",
        },
        "xsum": {
            "target_ds_label": "target",
            "target_prompt_label": "Summary:",
            "input_ds_label": "document",
            "input_prompt_label": "Document:",
        },
        "cochrane-simplification": {
            "target_ds_label": "target",
            "target_prompt_label": "Simplified:",
            "input_ds_label": "source",
            "input_prompt_label": "Abstract:",
        },
        "common_gen": {
            "target_ds_label": "target",
            "target_prompt_label": "Sentence:",
            "input_ds_label": "concepts",
            "input_prompt_label": "Concepts:",
        },
    }

    context_length = CONTEXT_LENGTH
    if dataset == "xsum" or dataset == "cochrane-simplification":
        context_length = 512

    def _test_dataset_tokenizer(element):
        targets = element[dataset_structure[dataset]["target_ds_label"]]
        inps = element[dataset_structure[dataset]["input_ds_label"]]
        outputs, references = [], []
        for target, inp in zip(targets, inps):
            if isinstance(inp, list):
                inp = " ".join(inp)
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']}"  # noqa: E501
            )
            references.append(target)
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt",
        )
        return {**tokenized, "references": references}

    def _gem_train_dataset_tokenizer(element):
        targets = element[dataset_structure[dataset]["target_ds_label"]]
        inps = element[dataset_structure[dataset]["input_ds_label"]]
        outputs = []
        for target, inp in zip(targets, inps):
            if isinstance(inp, list):
                inp = " ".join(inp)
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']} {target}"  # noqa: E501
            )
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt",
        )
        return tokenized

    tokenized_train = train_ds.map(
        _gem_train_dataset_tokenizer,
        batched=True,
        remove_columns=[
            col
            for col in train_ds.column_names
            if col not in ["input_ids", "attention_mask"]
        ],
    )
    tokenized_train.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_train, batch_size=train_batch_size, shuffle=True
    )

    tokenized_test = test_ds.map(
        _test_dataset_tokenizer,
        batched=True,
        remove_columns=[
            col
            for col in train_ds.column_names
            if col not in ["input_ids", "attention_mask", "references"]
        ],
    )
    tokenized_test.set_format("torch")
    test_dataloader = DataLoader(
        tokenized_test, batch_size=test_batch_size, shuffle=True
    )

    return train_dataloader, test_dataloader


def evaluate_gem_dataset(
    model, eval_dataloader, tokenizer, dataset_name, use_sampler=False
):
    results = []
    max_new_tokens = 25
    if dataset_name in ["xsum", "cochrane-simplification"]:
        max_new_tokens = 256
    for batch in tqdm(eval_dataloader):
        params = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampler:
            params = {
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
            }
        outputs = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **params,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i, output in enumerate(outputs):
            previous_tokens = tokenizer.decode(
                output[: len(batch["input_ids"][i])],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predicted_tokens = tokenizer.decode(
                output[len(batch["input_ids"][i]) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = gem_metrics.texts.Predictions([predicted_tokens])
            references = gem_metrics.texts.References([batch["references"][i]])
            rouge_results = gem_metrics.compute(
                predictions, references, metrics_list=["rouge"]
            )
            if i == 0 and ARGS.debug:
                logger.debug("Context", previous_tokens.strip())
                logger.debug("Predicted", predicted_tokens.strip())
                logger.debug("Correct", batch["references"][i].strip())
            rouge_results["predicted"] = predicted_tokens.strip()
            results.append(rouge_results)

    return {
        "rouge1": sum([result["rouge1"]["fmeasure"] for result in results])
        / len(results),
        "rouge2": sum([result["rouge2"]["fmeasure"] for result in results])
        / len(results),
        "rougeL": sum([result["rougeL"]["fmeasure"] for result in results])
        / len(results),
        "rouge1_per_sample": [result["rouge1"]["fmeasure"] for result in results],
        "rouge2_per_sample": [result["rouge2"]["fmeasure"] for result in results],
        "rougeL_per_sample": [result["rougeL"]["fmeasure"] for result in results],
        "generated_text": [result["predicted"] for result in results],
    }


class GEMDataset(BenchmarkDataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        test_batch_size=DEFAULT_BATCH_SIZE,
        train_batch_size=DEFAULT_BATCH_SIZE,
    ):
        self.tokenizer = tokenizer
        self.train_dataloader, self.test_dataloader = construct_gem_dataset(
            dataset_name, tokenizer, test_batch_size, train_batch_size
        )
        self._name = dataset_name

    @property
    def train_dataloader(self):
        return self.train_dataloader

    @property
    def test_dataloader(self):
        return self.test_dataloader

    @property
    def name(self):
        return self._name

    def evaluate(self, model, dataset_split: Literal["train", "test"] = "test"):
        if dataset_split == "train":
            return evaluate_gem_dataset(model, self.train_dataloader, self.tokenizer)
        return evaluate_gem_dataset(model, self.test_dataloader, self.tokenizer)
