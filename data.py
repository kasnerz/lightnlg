#!/usr/bin/env python3

import json
import csv
import os
import logging
import re
import random
import datasets

from collections import defaultdict, namedtuple

logger = logging.getLogger(__name__)


def get_dataset_class_by_name(name):
    """
    A helper function which provides mapping between dataset classes and command-line arguments.
    """
    try:
        dataset_mapping = {
            "scitldr": "ExampleHFDataset",
            "tiny_shakespeare": "ExampleCustomDataset",
        }
        dataset_module = __import__(
            __name__,
            globals=globals(),
            fromlist=[dataset_mapping[name]],
            level=0,
        )
        dataset_class = getattr(dataset_module, dataset_mapping[name])
        return dataset_class
    except KeyError:
        logger.error(
            f"Unknown dataset: '{name}'. Please create a class in 'data.py' and add the mapping to `data.py:get_dataset_class_by_name()`."
        )
        raise


class Dataset:
    """
    Base class for the datasets
    """

    def __init__(self):
        self.data = {split: [] for split in ["train", "dev", "test"]}

    def load(self, splits, path=None):
        """
        Load the dataset. Path can be specified for loading from a directory
        or omitted if the dataset is loaded from HF.
        """
        raise NotImplementedError


class ExampleHFDataset(Dataset):
    # source: https://huggingface.co/datasets/scitldr
    def __init__(self):
        super().__init__()

    def load(self, splits, path=None):
        """
        Load the dataset using HF `datasets`
        """
        dataset = datasets.load_dataset("scitldr")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                entry = (" ".join(example["source"]), " ".join(example["target"]))
                self.data[split].append(entry)


class ExampleCustomDataset(Dataset):
    # source: https://github.com/jcjohnson/torch-rnn/blob/master/data/tiny-shakespeare.txt
    def __init__(self):
        super().__init__()

    def load(self, splits, path=None):
        """
        Load the dataset from the input directory
        """
        block_size = 1024
        i = 0

        # split the contiguous text into blocks of size `max_input_length` for GPT-2
        with open(os.path.join(path, "input.txt")) as f:
            text = f.read()
            idx = 0

            while idx + block_size < len(text):
                block = text[idx : idx + block_size]

                # 8/1/1 train/dev/test splits
                if i % 10 == 0:
                    split = "test"
                elif i % 10 == 1:
                    split = "dev"
                else:
                    split = "train"

                self.data[split].append(block)
                idx += block_size
                i += 1
