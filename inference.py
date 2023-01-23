#!/usr/bin/env python3

import logging

from model import CausalLMTrainingModule, Seq2SeqTrainingModule
from model import add_special_tokens
from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class InferenceModule:
    def __init__(self, args, training_module_cls, model_path=None):
        self.args = args
        self.beam_size = args.beam_size
        self.special_tokens = training_module_cls.special_tokens

        if model_path is not None:
            self.model = training_module_cls.load_from_checkpoint(model_path)
            self.model.freeze()
            self.model_name = self.model.model.name_or_path
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model_name = args.model_name
            self.model = training_module_cls(args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def predict(self, s):
        inputs = self.tokenizer(s, return_tensors="pt")

        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            self.model.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        else:
            logger.warning("Not using GPU")

        return self.generate(inputs["input_ids"])

    def generate(self, input_ids):
        raise NotImplementedError


class CausalLMInferenceModule(InferenceModule):
    def __init__(self, args, model_path=None):
        super().__init__(args, model_path=model_path, training_module_cls=CausalLMTrainingModule)

    def generate(self, input_ids):
        # top-k sampling, other methods TBD
        out = self.model.model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            max_length=self.args.max_length,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        sentences = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return sentences


class Seq2SeqInferenceModule(InferenceModule):
    def __init__(self, args, model_path=None):
        super().__init__(args, model_path=model_path, training_module_cls=Seq2SeqTrainingModule)

    def generate(self, input_ids):
        out = self.model.model.generate(
            input_ids,
            max_length=self.args.max_length,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
        )
        sentences = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return sentences
