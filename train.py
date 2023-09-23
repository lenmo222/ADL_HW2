import json
import torch
import datasets
import argparse
import collections
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import chain
from datasets import Dataset
from dataclasses import dataclass
from typing import Optional, Union
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from transformers import default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModelForQuestionAnswering

# preproccessing function multiple choice
def preprocess_function(examples):
    # sent1 4 times repeat(all)
    first_sentences = [[q] * 4 for q in examples["question"]]
    question_headers = examples["paragraphs"]
    second_sentences = [[context[i[j]] for j in range(0,4)]for i in question_headers]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True,padding='max_length',max_length=512)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

# preproccessing function QA
def preprocess_function_QA(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer_QA(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["start"][0]
        end_char = answer["start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def compute_metrics(eval_predictions):
    predictions, labels_ids = eval_predictions
    preds = np.argmax(predictions,axis=-1)
    return {"accuracy":(preds == labels_ids).astype(np.float32).mean().item()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        if(label_name=="label"):
            labels = [feature.pop(label_name) for feature in features]
        else :
            labels = [feature for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            # padding=self.padding,
            # max_length=self.max_length,
            max_length=512,
            padding='max_length',
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def parser():
    parser = argparse.ArgumentParser(description="ADL HW2 file")
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--eval_file", type=str, default="")
    parser.add_argument("--context_file", type=str, default="")
    return parser.parse_args()

args = parser()
if args.train_file:
    with open(args.train_file,encoding="utf-8") as f:
        data_train = json.load(f)
        data_train = pd.DataFrame(data_train)
if args.eval_file:
    with open(args.eval_file,encoding="utf-8") as f:
        data_val = json.load(f)
        data_val = pd.DataFrame(data_val)
if args.context_file:
    with open(args.context_file,encoding="utf-8") as f:
        context = json.load(f)

# add label to multiple choice
labels_train = []
labels_valid = []
for i in range(0,len(data_train['paragraphs'])):
    for j in range(0,4):
        if(data_train['relevant'][i]==data_train['paragraphs'][i][j]):
            labels_train.append(j)
            break
for i in range(0,len(data_val['paragraphs'])):
    for j in range(0,4):
        if(data_val['relevant'][i]==data_val['paragraphs'][i][j]):
            labels_valid.append(j)
            break
data_train['label'] = labels_train
data_val['label'] = labels_valid
train = Dataset.from_pandas(data_train)
val = Dataset.from_pandas(data_val)
data = datasets.DatasetDict({"train":train,"val":val})

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
tokenized_swag = data.map(preprocess_function, batched=True)

model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-xlnet-base")

# set model to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# multiple choice train
training_args = TrainingArguments(
    output_dir="./results_multiple_choice",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.01,
    adam_beta1=0.9
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["val"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# save model
model_dir = 'mupile_model_8'
trainer.save_model(model_dir)

if args.train_file:
    with open(args.train_file,encoding="utf-8") as f:
        data_train_QA = json.load(f)
        data_train_QA = pd.DataFrame(data_train_QA)
if args.eval_file:
    with open(args.eval_file,encoding="utf-8") as f:
        data_val_QA = json.load(f)
        data_val_QA = pd.DataFrame(data_val_QA)
if args.context_file:
    with open(args.context_file,encoding="utf-8") as f:
        context = json.load(f)

# add context to QA
context_train_QA = []
context_val_QA = []
for i in range(0,len(data_train_QA['relevant'])):
    context_train_QA.append(context[data_train_QA['relevant'][i]])
    data_train_QA['answer'][i]['text'] = [data_train_QA['answer'][i]['text']] 
    data_train_QA['answer'][i]['start'] = [data_train_QA['answer'][i]['start']]
for i in range(0,len(data_val_QA['relevant'])):
    context_val_QA.append(context[data_val_QA['relevant'][i]])
    data_val_QA['answer'][i]['text'] = [data_val_QA['answer'][i]['text']]
    data_val_QA['answer'][i]['start'] = [data_val_QA['answer'][i]['start']]
data_train_QA['context'] = context_train_QA
data_val_QA['context'] = context_val_QA
train_QA = Dataset.from_pandas(data_train_QA)
val_QA = Dataset.from_pandas(data_val_QA)
data_QA = datasets.DatasetDict({"train":train_QA,"val":val_QA})

tokenizer_QA = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenized_squad = data_QA.map(preprocess_function_QA, batched=True, remove_columns=data_QA["train"].column_names)

data_collator_QA = DefaultDataCollator()
model_QA = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-roberta-wwm-ext")

# model QA to cuda
model_QA.to(device)

# train QA model
training_args_QA = TrainingArguments(
    output_dir="./ADL_HW2_QA_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    gradient_accumulation_steps=2
)

trainer_QA = Trainer(
    model=model_QA,
    args=training_args_QA,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["val"],
    tokenizer=tokenizer_QA,
    data_collator=data_collator_QA,
)

trainer_QA.train()

# save QA model
model_dir_QA = 'QA_4'
model_dir_QA.save_model(model_dir_QA)