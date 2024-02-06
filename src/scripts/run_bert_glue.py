# This is a MWE on how to use the LDRM-PEFT models. One can use the HF trainers or this boiler plate code. Simply import the correct model. 

import argparse
import os
import logging
import json

import numpy as np
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig, PretrainedConfig
from tqdm import tqdm

from adapter.circulant_adapter import CustomBertForSequenceClassification
from utils.utils import mark_only_circulant_as_trainable


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


parser = argparse.ArgumentParser(
    description="Circulant Finetuning on GLUE on 1 GPU"
)
parser.add_argument("--seed", default=3047, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
parser.add_argument("--task_name", default='rte', type=str) 
parser.add_argument("--device", default='cuda')
parser.add_argument("--num_epochs", default=80, type=int)
parser.add_argument("--lr", default=2e-3, type=float)
parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )
parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
parser.add_argument("--weight_decay", type=float, default=0.25, help="Weight decay to use.")




args = parser.parse_args(args=[])
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

raw_datasets = load_dataset("glue", args.task_name)

if args.task_name is not None:
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
        # some defaults et here
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
print(f"model used is {args.model_name_or_path}")

config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='right')
model = CustomBertForSequenceClassification(config, circ_dropout=.15, activation="gelu", use_prod=True)

if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


 # Preprocessing the datasets
if args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
else:
      # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

  # Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
      model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
      and args.task_name is not None
      and not is_regression
  ):
      # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        logger.info(
              f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
              "Using it!"
          )
        label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
    else:
        logger.warning(
              "Your model seems to have been trained with labels, but they don't match the dataset: ",
              f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
              "\nIgnoring the model labels as a result.",
          )
elif args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif args.task_name is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

padding = "max_length" if args.pad_to_max_length else False

mark_only_circulant_as_trainable(model)
for n,p in model.named_parameters():
  if 'classifier' in n: # update classifier params as well
    p.requires_grad = True

def preprocess_function(examples):
        # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
                # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result


processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

if args.task_name is not None:
    metric = evaluate.load("glue", args.task_name)
else:
    metric = evaluate.load("accuracy")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)


eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
) 
if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
    test_dataset = processed_datasets["validation_mismatched"]
else :
    test_dataloader = DataLoader(
        processed_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)


# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs), # kinda hardcoded. 
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

model.to(args.device)

epochs = 0
best_acc = o

d = {}
for epoch in range(epochs, args.num_epochs):
    model.train()
    loss_epoch = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(args.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss_epoch += loss.cpu().detach().numpy()
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("All summed loss for epoch is ", loss_epoch)

    epoch = epoch + 1
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    d[epoch] = eval_metric



