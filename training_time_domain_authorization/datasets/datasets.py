from abc import ABC
from typing import Literal

from training_time_domain_authorization.datasets.gem import GEMDataset


class BenchmarkDataset(ABC):
    """BenchmarkDataset is an abstract class
    that defines the interface for all benchmark datasets.
    It must implement the following:
    - a constructor that takes in dataset construction parameters
    - a training and testing pytorch dataloader (validation is optional)
    - an evaluation function that takes in a HF model and returns a score
    - - evaluation function must output a per sample test metrics as well as generated text for each test sample
    - - This is because we cannot do statistical significance testing without per sample test metrics
    - - This is because we cannot do manual inspection without generated text
    """

    def __init__(self, **kwargs):
        pass

    @property
    def train_dataloader(self):
        raise NotImplementedError

    @property
    def test_dataloader(self):
        raise NotImplementedError

    @property
    def validation_dataloader(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def evaluate(self, model, dataset: Literal["train", "test", "validation"] = "test"):
        """Evaluate the given huggingface CausalLM model on the dataset
        using the dataset specific metrics
        """
        raise NotImplementedError


def construct_dataset(dataset_name, tokenizer, test_batch_size, train_batch_size):
    """Construct a benchmark dataset given the dataset name"""
    if dataset_name in ["viggo"]:
        return GEMDataset(dataset_name, tokenizer, test_batch_size, train_batch_size)
    else:
        raise NotImplementedError
