import argparse

parser = argparse.ArgumentParser(description="TTDA Experiment Runner")
# experiment arguments
parser.add_argument(
    "--experiment-name", type=str, required=True, help="name of experiment"
)
parser.add_argument("--dataset", type=str, default="viggo", help="dataset to use")
parser.add_argument("--model", type=str, default="distilgpt2", help="model to use")
parser.add_argument(
    "--tokenizer", type=str, default="distilgpt2", help="tokenizer to use"
)
parser.add_argument(
    "--save-results", type=bool, default=True, help="save results to file"
)
parser.add_argument("--debug", type=bool, default=False, help="log debug messages")

# general training arguments / hyperparameters
parser.add_argument(
    "--train-batch-size", type=int, default=8, help="batch size for training"
)
parser.add_argument(
    "--test-batch-size", type=int, default=8, help="batch size for testing"
)
parser.add_argument(
    "--num-epochs", type=int, default=1, help="number of epochs to train for"
)
parser.add_argument(
    "--learning-rate", type=float, default=1e-5, help="learning rate for training"
)
parser.add_argument(
    "--steps-to-eval", type=int, default=100, help="steps to evaluate model"
)

# specifici hyperparameters
parser.add_argument("--seed", type=int, default=42, help="random seed")

ARGS = parser.parse_args()
