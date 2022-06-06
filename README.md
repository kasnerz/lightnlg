# PyTorch Lightning NLG Starter Pack

A basic set of scripts for training NLG models from [HuggingFace Transformers](https://huggingface.co/docs/transformers) using [PyTorch Lightning](https://www.pytorchlightning.ai).

The code can be used as-is for (1) language modeling using GPT-2 and (2) sequence-to-sequence generation using BART.

The repository contains the following files:
- `data.py`- Classes for loading external datasets. Contains examples for a loading a HuggingFace dataset and custom CSV-based dataset.
- `preprocess.py` - Preprocessing external datasets into JSON files.
- `dataloader.py`- Transforming preprocessed JSON datasets into PyTorch Dataloaders.
- `model.py` - Model-specific code for training.
- `inference.py` - Model-specific code for inference (decoding algorithms etc.).
- `train.py` - Running training.
- `decode.py` - Running inference on val / test splits.
- `interact.py` - Running inference using user input.

