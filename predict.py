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

def preprocess_function(examples):
    # sent1 4 times repeat(all)
    first_sentences = [[q] * 4 for q in examples["question"]]
    question_headers = examples["paragraphs"]
    second_sentences = [[context[i[j]] for j in range(0,4)]for i in question_headers]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True,padding='max_length',max_length=512)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from itertools import chain

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
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            max_length=512,
            padding='max_length',
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch

def parser():
    parser = argparse.ArgumentParser(description="ADL HW2 predict file")
    parser.add_argument("--context_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--pred_file", type=str, default="")
    return parser.parse_args()

args = parser()
if args.context_file:
    with open(args.context_file,encoding="utf-8") as f:
        context = json.load(f)
if args.test_file:
    with open(args.test_file,encoding="utf-8") as f:
        data_test = json.load(f)
        data_test = pd.DataFrame(data_test)

test = Dataset.from_pandas(data_test)
data = datasets.DatasetDict({"test":test})

model_checkpoint = 'mupile_model_8'
# multiple choice load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_swag = data.map(preprocess_function, batched=True)
# load multiple choice model
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
# multiple choice model to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# predict multiple choice
keys = ["input_ids","attention_mask"]
features = [{k: v for k, v in tokenized_swag["test"][i].items() if k in keys} for i in range(len(tokenized_swag["test"]["id"]))]
batch = DataCollatorForMultipleChoice(tokenizer)(features)
predict_array = []
for i in range(0,len(batch["input_ids"])):
    input_loop = [batch["input_ids"][i][j].tolist() for j in range(4)]
    inputs_loop_ids = torch.tensor(input_loop).unsqueeze(0).to(device)
    labels_loop = torch.tensor(1).unsqueeze(0).to(device)
    outputs_loop = model(inputs_loop_ids,labels=labels_loop)
    outputs_loop = outputs_loop['logits'].tolist()[0]
    predict_array.append(outputs_loop.index(max(outputs_loop)))

data_test_csv = data_test
answer_context = []
for i in range(len(predict_array)):
    answer_context.append(context[data_test_csv["paragraphs"][i][predict_array[i]]])
data_test_csv['answer_index'] = predict_array
data_test_csv['answer_context'] = answer_context
# save multiple choice predict
data_test_csv.to_csv("choice_predict(no_final_predict).csv")

if args.context_file:
    with open(args.context_file,encoding="utf-8") as f:
        context = json.load(f)
if args.test_file:
    with open(args.test_file,encoding="utf-8") as f:
        data_test_QA = json.load(f)
        data_test_QA = pd.DataFrame(data_test_QA)

data_test_context = pd.read_csv("choice_predict(no_final_predict).csv")
data_test_QA['context'] = data_test_context['answer_context']

test_QA = Dataset.from_pandas(data_test_QA)
dataDict_test = datasets.DatasetDict({"test":test_QA})

model_checkpoint_QA = 'QA_4'
tokenizer_QA = AutoTokenizer.from_pretrained(model_checkpoint_QA)

def preprocess_test_QA(examples):
    inputs = tokenizer_QA(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["example_id"] = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        inputs["example_id"].append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset_mapping[i] = [(o if s == 1 else None) for o, s in zip(offset, sequence_ids)]

    return inputs

test_features = dataDict_test["test"].map(
    preprocess_test_QA,
    batched=True,
    remove_columns=dataDict_test["test"].column_names,
)
model_QA = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint_QA).to(device)
dataloader = DataLoader(
    test_features.remove_columns(["example_id", "offset_mapping"]),
    batch_size=64,
    collate_fn=default_data_collator
)

start_logits = []
end_logits = []
for batch in tqdm(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model_QA(**batch)
    start_logits.append(outputs.start_logits.cpu())
    end_logits.append(outputs.end_logits.cpu())
start_logits = torch.cat(start_logits, dim=0).numpy()
end_logits = torch.cat(end_logits, dim=0).numpy()

example_to_feature = collections.defaultdict(list)
for idx, feature in enumerate(test_features):
    example_id = feature["example_id"]
    example_to_feature[example_id].append(idx)

predicted_answers = {}
for example in tqdm(dataDict_test["test"]):
    example_id = example["id"]
    context = example["context"]
    answers = []
    n_best_size = 20
    max_answer_length = 30
    for feature_index in example_to_feature[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = test_features[feature_index]["offset_mapping"]

        start_indexes = np.argsort(start_logit)[-1 : -n_best_size : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best_size : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Predicting (0, 0) means no answer.
                if start_index == 0 and end_index == 0:
                    answers.append({"text": "", "logit_score": start_logit[start_index] + end_logit[end_index]})
                # Skip answers that are not fully in the context.
                elif offsets[start_index] is None or offsets[end_index] is None:
                    continue
                elif len(offsets[start_index])==0 or len(offsets[end_index])==0:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length.
                elif start_index >= len(offsets) or end_index >= len(offsets):
                    continue
                elif end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                else:
                    answers.append({
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })

    best_answer = max(answers, key= lambda x: x["logit_score"])
    predicted_answers[example_id] = best_answer["text"]

answers_array = []
for i in predicted_answers:
    answers_array.append(predicted_answers[i])

data_test_output = pd.DataFrame()
data_test_output["id"] = data_test_QA["id"]
data_test_output["answer"] = answers_array
data_test_output = data_test_output.set_index("id")

if args.pred_file:
    data_test_output.to_csv(args.pred_file)
else:
    data_test_output.to_csv('prediction.csv')