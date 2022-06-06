# LightNLG: A Starter Pack for NLG with PyTorch Lightning :zap: 

A minimalistic codebase for training natural language generation models from [HuggingFace Transformers](https://huggingface.co/docs/transformers) using [PyTorch Lightning](https://www.pytorchlightning.ai).

The code can be used as-is for:
1) **language modeling** using an autoregressive LM (e.g, GPT-2),
2)  **sequence-to-sequence generation** using an encoder-decoder (e.g, BART).

You can get started ASAP using the examples below which include:
- **interacting** with vanilla / finetuned GPT-2 and BART,
- **finetuning GPT-2** on the [Tiny Shakespeare](https://github.com/jcjohnson/torch-rnn/blob/master/data/tiny-shakespeare.txt) dataset,
- **finetuning BART** on the [SciTLDR](https://huggingface.co/datasets/scitldr) dataset (TL;DR from abstracts).

Feel free to use the codebase as a skeleton for more advanced data processing, model tweaks, etc.

Overview of scripts:
- `data.py`- Loading external datasets (contains examples of a HuggingFace dataset and a custom plaintext dataset).
- `preprocess.py` - Preprocessing external datasets into JSON files.
- `dataloader.py`- Transforming preprocessed JSON datasets into PyTorch Dataloaders.
- `model.py` - Model-specific code for training.
- `inference.py` - Model-specific code for inference (decoding algorithms etc.).
- `train.py` - Running training.
- `decode.py` - Running batch inference on dev / test splits.
- `interact.py` - Running interactive inference with user input.

## Quickstart
```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Examples: Language Modeling

### :writing_hand: Interactive Prompting with GPT-2
```
./interact.py \
    --model_name gpt2  \
    --mode causal_lm \
    --max_length 200
```

```
[In]: Good morning!
[Out]:
['Good morning! I hope you all enjoyed my breakfast in the café. We are '
 'working with the media and I will provide more updates about the situation '
 'in your home and our working plans after the election, as well as an update '
 'on where our progress is going. My name is Richard Coughlin and I was in '
 'your office for our first business meeting.\n' (...)]
```

### :book: Finetuning GPT-2 on Tiny Shakespeare
```
./preprocess.py \
    --dataset "tiny_shakespeare"  \
    --dataset_dir "data/orig/tiny_shakespeare" \
    --mode causal_lm \
    --output "data/tiny_shakespeare"
```
```
./train.py \
    --mode causal_lm \
    --in_dir data/tiny_shakespeare \
    --experiment tiny_shakespeare \
    --gpus 1 \
    --model_name gpt2 \
    --accumulate_grad_batches 4 \
    --learning_rate 5e-4 \
    --max_epochs 5
```

### :feather: Generating Shakespeare using GPT-2
*Note: first finetune the model using the instructions above.*
```
./interact.py \
    --experiment tiny_shakespeare \
    --mode causal_lm \
    --max_length 200
```
**Example output**
```
[In]: Good morning! 
[Out]:
['Good morning! \n'
 '\n'
 'PETRUCHIO:\n'
 'And thou shalt have a father till she speak,\n'
 "For my son's sake be ready,\n"
 "I charge thee, be thou ready at five o'clock.\n" (...)]
```


## Examples: Seq2Seq Generation

### :mag: Interactive Denoising with BART
```
./interact.py \
    --model_name facebook/bart-base  \
    --mode seq2seq \
    --max_length 200 \
    --beam_size 3
```

**Example output**
```
[In]: This sentence is a bit <mask>.
[Out]:
['This sentence is a bit long.', 'This sentence is a bit of a stretch.', 'This sentence is a bit of a mess.']
```


### :microscope: Finetuning BART on SciTLDR
```
./preprocess.py \
    --dataset "scitldr"  \
    --mode seq2seq \
    --output "data/scitldr"
```
```
./train.py \
    --mode seq2seq \
    --in_dir data/scitldr \
    --experiment scitldr \
    --gpus 1 \
    --model_name facebook/bart-base \
    --accumulate_grad_batches 4 \
    --max_epochs 5
```

### :zap: Generating TD;DR from Scientific Paper Abstracts
*Note: first finetune the model using the instructions above.*

**Batch decoding**
```
./decode.py \
    --experiment "scitldr" \
    --in_dir data/scitldr \
    --split test \
    --gpus 1 \
    --out_filename test.out
```

**Interactive**
```
./interact.py \
    --experiment scitldr \
    --mode seq2seq \
    --max_length 200 \
    --beam_size 3
```
**Example output**
```
[In]: In data-to-text (D2T) generation, training on in-domain data leads to overfitting to the data representation and repeating training data noise. We examine how to avoid finetuning the pretrained language models (PLMs) on D2T generation datasets while still taking advantage of surface realization capabilities of PLMs. Inspired by pipeline approaches, we propose to generate text by rephrasing single-item templates using a sequence of modules trained on general-domain text-based operations—ordering, aggregation, and paragraph compression. We train PLMs for performing these operations on a synthetic corpus WikiFluent which we build from English Wikipedia. Our experiments on two major triple-to-text datasets—WebNLG and E2E—show that our approach enables D2T generation from RDF triples in zero-shot settings.
[Out]:
['We propose to avoid finetuning the pretrained language models (PLMs) on D2T generation datasets while still taking advantage of surface realization capabilities of PLMs.',
 'We propose to avoid finetuning the pretrained language models on D2T generation datasets while still taking advantage of surface realization capabilities of PLMs.',
 'We train PLMs to generate text by rephrasing single-item templates using a sequence of modules trained on general-domain text-based operations.']

```