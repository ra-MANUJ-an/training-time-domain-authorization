import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer

from loguru import logger

from training_time_domain_authorization.arguments import ARGS
from training_time_domain_authorization.datasets.datasets import construct_dataset
from training_time_domain_authorization.training import train_model

args = ARGS
logger.info(f"Running experiment: {args.experiment_name}")
print(args)

if __name__ == "__main__":
    # set all seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_name = args.model
    if args.local_model:
        model_name = f"./models/{args.local_model}"

    # check if ./results/{args.experiment_name}.json already exists if so exit
    if os.path.exists(f"./results/{args.experiment_name}.json"):
        logger.info(f"Experiment {args.experiment_name} already exists, exiting")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = construct_dataset(
        args.dataset,
        tokenizer,
        test_batch_size=args.test_batch_size,
        train_batch_size=args.train_batch_size,
    )
    # train model
    model, results = train_model(model_name, dataset)
    # save model to local
    logger.info("Saving trained model and results")
    model_name = f"{args.experiment_name}"
    if args.save_model:
        model.save_pretrained(f"./models/{model_name}")
    if args.save_results:
        # save results
        with open(f"./results/{args.experiment_name}.json", "w") as f:
            json.dump(results, f)
        # save args
        with open(f"./results/{args.experiment_name}_params.json", "w") as f:
            json.dump(vars(args), f)
