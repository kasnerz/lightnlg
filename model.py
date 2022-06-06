#!/usr/bin/env python3

import numpy as np
import os
import logging
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer,
    get_scheduler
)
from torch.optim import (
    AdamW,
    Adagrad
)
logger = logging.getLogger(__name__)


class TrainingModule(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["datamodule"])
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                       use_fast=True)
        self.datamodule = kwargs.get("datamodule", None)


    def forward(self, **inputs):
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        return {"loss": out["loss"], "logits": out["logits"]}


    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log('loss/train', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log('loss/val', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )
        total_steps = self.args.max_steps if (self.args.max_steps is not None and self.args.max_steps != -1) else len(
            self.datamodule.train_dataloader()) * self.args.max_epochs
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "polynomial",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"Using Adam optimizer")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)

        # these should be reasonable defaults for the seq2seq models
        # for finetuning GPT-2,you can use higher learning rate (e.g., 5e-4)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser


class CausalLMTrainingModule(TrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            return_dict=True
        )


class Seq2SeqTrainingModule(TrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            return_dict=True
        )

    def test_step(self, batch, batch_idx):
        out = self.model.generate(batch["input_ids"], 
            max_length=self.args.max_length,
            num_beams=1,
            num_return_sequences=1)
        
        out = self.tokenizer.batch_decode(out, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # log the generated output and write it in the output file
        for idx, o in enumerate(out):
            logger.info(f"[{batch_idx * len(out) + idx}] {o}")
            self.out_file_handle.write(o + "\n")