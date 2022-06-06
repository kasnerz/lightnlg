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
    try:
        # case-insensitive
        available_classes = {o.name.lower() : o for o in globals().values() 
                                if type(o)==type(Dataset) and hasattr(o, "name")}
        return available_classes[name.lower()]
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create \
            a class with an attribute name='{args.dataset}' in 'data.py'.")
        return None


class Dataset:
    def __init__(self):
        self.data = {split: [] for split in ["train", "dev", "test"]}

    def load(self, splits, path=None):
        """
        Load the dataset. Path can be specified for loading from a directory.
        """
        raise NotImplementedError


class ExampleHFDataset(Dataset):
    pass
    # name = "openwebtext-10k"

    # def __init__(self):
    #     super().__init__()

    # def load(self, splits, path=None):
    #     """
    #     Load the dataset using HF `datasets`
    #     """
    #     dataset = datasets.load_dataset("stas/openwebtext-10k")

    #     for i, example in enumerate(dataset["train"]):
    #         text = example["text"]

    #         if i % 10 == 0:
    #             split = "test"
    #         elif i % 10 == 1:
    #             split = "dev"
    #         else:
    #             split = "train"

    #         self.data[split].append(text)


class ExampleCustomDataset(Dataset):
    name = "tiny_shakespeare"

    def __init__(self):
        super().__init__()

    def load(self, splits, path=None):
        """
        Load the dataset
        """
        block_size = 1024
        i = 0

        with open(os.path.join(path, "input.txt")) as f:
            text = f.read()
            idx = 0

            while idx + block_size < len(text):
                block = text[idx:idx+block_size]

                if i % 10 == 0:
                    split = "test"
                elif i % 10 == 1:
                    split = "dev"
                else:
                    split = "train"

                self.data[split].append(block)
                idx += block_size
                i += 1