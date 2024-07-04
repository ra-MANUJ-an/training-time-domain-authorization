from training_time_domain_authorization.datasets.gem import GEMDataset


def construct_dataset(dataset_name, tokenizer, test_batch_size, train_batch_size):
    """Construct a benchmark dataset given the dataset name"""
    if dataset_name in ["viggo"]:
        return GEMDataset(dataset_name, tokenizer, test_batch_size, train_batch_size)
    else:
        raise NotImplementedError
