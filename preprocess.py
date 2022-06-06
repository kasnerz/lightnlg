#!/usr/bin/env python3

import os
import argparse
import logging
import data
import json
import random
import re
import numpy as np
from collections import defaultdict
from data import get_dataset_class_by_name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, dataset, out_dirname, mode):
        self.dataset = dataset
        self.out_dirname = out_dirname
        self.mode = mode

    def create_examples(self, entry, dataset):
        """
        Generates training examples from an entry in the dataset
        """
        examples = []

        if self.mode == "causal_lm":
            # input == output for causal LM
            example = {
                "in" : entry
            }
        elif self.mode == "seq2seq":
            # a simple case without any processing
            example = {
                "in" : entry[0],
                "out" : entry[1]
             }
        else:
            raise ValueError("Unknown mode")

        examples.append(example)
        return examples


    def process(self, split):
        output = {"data" : []}
        data = self.dataset.data[split]

        for i, entry in enumerate(data):
            examples = self.create_examples(entry, dataset)

            for example in examples:
                output["data"].append(example)

        with open(os.path.join(self.out_dirname, f"{split}.json"), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="Name of the dataset to preprocess.")
    parser.add_argument("--dataset_dir", type=str, default=None,
        help="Path to the dataset")
    parser.add_argument("--mode", choices=["causal_lm", "seq2seq"], required=True,
        help="Preprocessing mode, depends on the dataset")
    parser.add_argument("--output", type=str, required=True,
        help="Name of the output directory")
    parser.add_argument('--splits', type=str, nargs='+', default=["train", "dev", "test"],
                    help='Dataset splits (e.g. train dev test)')
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")

    args = parser.parse_args()
    random.seed(args.seed)
    logger.info(args)
    dataset = get_dataset_class_by_name(args.dataset)()

    try:
        dataset.load(splits=args.splits, path=args.dataset_dir)
    except FileNotFoundError as err:
        logger.error(f"Dataset could not be loaded")
        raise err
        
    try:
        out_dirname = args.output
        os.makedirs(out_dirname, exist_ok=True)
    except OSError as err:
        logger.error(f"Output directory {out_dirname} can not be created")
        raise err

    preprocessor = Preprocessor(
        dataset=dataset, 
        out_dirname=out_dirname,
        mode=args.mode
    )
    for split in args.splits:
        preprocessor.process(split)

    logger.info(f"Preprocessing finished.")