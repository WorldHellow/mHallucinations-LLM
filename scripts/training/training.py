import argparse
import json
import numpy as np
from datasets import DatasetDict, Dataset
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from billm import LlamaForTokenClassification, MistralForTokenClassification
import torch
import wandb
from accelerate import PartialState
import os
import logging
import random
from sklearn.metrics import classification_report
import sys 
import gzip

logging.basicConfig(
    filename="tokenize_debug.log", level=logging.DEBUG, format="%(message)s"
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="meta-llama/Meta-Llama-3-8B",
    help="Specify model_name_or_path to set transformer backbone."
)
parser.add_argument(
    "--train_file",
    type=str,
    default="train_binary.json",
    help="Specify the path to the training dataset file."
)
parser.add_argument(
    "--test_file",
    type=str,
    default="test_binary.json",
    help="Specify the path to the testing dataset file."
)
parser.add_argument(
    "--mode", type=str, default="bio_tags_binary", help="bio_tags_binary or tags"
)
parser.add_argument(
    "--language",
    type=str,
    default="urdu",
    help="Specify language in the train and test sets."
)
parser.add_argument(
    "--epochs",
    type=int,
    default=2,
    help="Specify number of epochs, default 50"
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="Specify number of batch size, default 8"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Specify learning rate, default 1e-4"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Specify weight decay, default 0.01"
)
parser.add_argument(
    "--max_length", type=int, default=4096, help="Specify max length, default 64"
)
parser.add_argument(
    "--lora_r", type=int, default=32, help="Specify lora r, default 32"
)
parser.add_argument(
    "--lora_alpha", type=int, default=32, help="Specify lora alpha, default 32"
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.05,
    help="Specify lora dropout, default 0.05"
)
# configure hub
parser.add_argument(
    "--push_to_hub",
    type=int,
    default=0,
    choices=[0, 1],
    help="Specify push_to_hub, default 0"
)
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="Specify push_to_hub_model_id, default None, format like organization/model_id"
)
parser.add_argument(
    "--seed_number",
    type=int,
    default=42,
    help="Specify seed number. Defaults to 42."
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default='Hallucination_Classifier_Category',
    help="Specify wandb project name."
)
args = parser.parse_args()
print(f"Args: {args}")

torch.manual_seed(args.seed_number)
random.seed(args.seed_number)
np.random.seed(args.seed_number)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed_number)
mode = args.mode
LANGUAGE = args.language

device_string = PartialState().process_index

# Your label2id mapping
if "binary" in mode:
    label2id = {"O": 0, "I-hallucination": 1, "ignore": -100}
    save_directory = "."
else:
    label2id = {
        "I-entity": 1,
        "I-relation": 2,
        "I-invented": 3,
        "I-contradictory": 4,
        "I-unverifiable": 5,
        "I-subjective": 6,
        "O": 0,
        "ignore": -100,
    }
    save_directory = "."

id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())


if "binary" in args.mode:
    save_mode = "Binary"
else:
    save_mode = "Category"

if "baseline" in args.wandb_project.lower():
    output_dir = os.path.join(
    save_directory, f"{save_mode}_{LANGUAGE}_seed{args.seed_number}_billm_ckpt_baseline"
    )
    final_model_dir = os.path.join(
        save_directory, f"Classifier_{save_mode}_{LANGUAGE}_seed{args.seed_number}_baseline"
    )

else:
    output_dir = os.path.join(
        save_directory, f"Big_{save_mode}_{LANGUAGE}_seed{args.seed_number}_billm_ckpt"
    )
    final_model_dir = os.path.join(
        save_directory, f"Big_Classifier_{save_mode}_{LANGUAGE}_seed{args.seed_number}"
    )

# Check if final model directory exists
if os.path.exists(final_model_dir):
    print(f"Final model directory {final_model_dir} exists. Training is already completed. Exiting.")
    sys.exit()

# Load the datasets
def load_json_dataset(train_path, test_path, language="all"):
    """
    Load and preprocess training and test datasets from JSON files.

    Args:
        train_path (str): Path to the training data JSON file.
        test_path (str): Path to the test data JSON file.
        language (str, optional): Language to filter by. Use "all" to include all languages. Defaults to "all".

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing "train" and "test" datasets.
    """

    def filter_language(data, lang, max_samples=None):
        """
        Filter the dataset for a specific language and optionally limit the number of samples.

        Args:
            data (list): The dataset to filter.
            lang (str): The target language to filter by.
            max_samples (int, optional): Maximum number of samples to include. Defaults to None.

        Returns:
            dict: A dictionary containing filtered tokens, tags, and language.
        """
        filtered_data = {"tokens": [], "tags": [], "language": []}
        count = 0
        for item in data:
            if item.get("language", "").lower() == lang.lower():
                filtered_data["tokens"].append(item.get("tokens", []))
                try:
                    filtered_data["tags"].append([label2id[label] for label in item.get(mode, [])])
                except KeyError as e:
                    raise ValueError(f"Label '{e.args[0]}' not found in label2id mapping.")
                filtered_data["language"].append(item.get("language", ""))
                count += 1
                if max_samples and count >= max_samples:
                    break
        return filtered_data

    print("Reading Training Data....")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        random.shuffle(train_data)

    print("Reading Test Data....")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        random.shuffle(test_data)

    if language.lower() != "all":
        # Filter training and test data for the specified language
        train_data_dict = filter_language(train_data, language)
        test_data_dict = filter_language(test_data, language, max_samples=50)
    else:
        # Include all training samples
        train_data_dict = {"tokens": [], "tags": [], "language": []}
        for item in train_data:
            lang = item.get("language", "")
            if not lang:
                continue  # Skip if language is missing
            train_data_dict["tokens"].append(item.get("tokens", []))
            try:
                train_data_dict["tags"].append([label2id[label] for label in item.get(mode, [])])
            except KeyError as e:
                raise ValueError(f"Label '{e.args[0]}' not found in label2id mapping.")
            train_data_dict["language"].append(lang)

        # Prepare test data with 100 samples per language
        test_data_dict = {"tokens": [], "tags": [], "language": []}
        test_language_counts = defaultdict(int)

        for item in test_data:
            lang = item.get("language", "")
            if not lang:
                continue  # Skip if language is missing
            if test_language_counts[lang] >= 100:
                continue  # Skip if limit reached for this language
            try:
                tags = [label2id[label] for label in item.get(mode, [])]
            except KeyError as e:
                raise ValueError(f"Label '{e.args[0]}' not found in label2id mapping.")
            if not tags:
                continue  # Skip if no tags
            test_data_dict["tokens"].append(item.get("tokens", []))
            test_data_dict["tags"].append(tags)
            test_data_dict["language"].append(item.get("language", ""))
            test_language_counts[lang] += 1

    # Create DatasetDict
    train_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(train_data_dict),
            "test": Dataset.from_dict(test_data_dict),
        }
    )

    # Display Dataset Statistics
    print(f"Number of training samples: {len(train_dataset['train'])}")
    unique_train_labels = {label_id for sublist in train_data_dict["tags"] for label_id in sublist}
    print(f"Unique labels in train set: {unique_train_labels}")

    print(f"Number of test samples: {len(train_dataset['test'])}")
    unique_test_labels = {label_id for sublist in test_data_dict["tags"] for label_id in sublist}
    print(f"Unique labels in test set: {unique_test_labels}")

    return train_dataset


print("\n##### Loading Data ##### ")
ds = load_json_dataset(args.train_file, args.test_file, LANGUAGE)
print(f"Loaded {LANGUAGE} Data.\n")
print(f"\n##### Loading Tokenizer {args.model_name_or_path} ##### \n")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir= None
)

if "mistral" in args.model_name_or_path.lower():
    tokenizer.add_special_tokens({"pad_token": "<unk>"})
if "llama" in args.model_name_or_path.lower():
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="longest",
        max_length=args.max_length,
        truncation=True,
    )

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    # Write a sample of tokenized_inputs['labels'][0] to a separate file
    with open("tokenized_labels_sample.txt", "w") as sample_file:
        sample_file.write(
            f"Sample tokenized_inputs['labels'][0]: {tokenized_inputs['labels'][0]}\n"
        )
        sample_file.write(
            f"Sample tokenized_inputs['input_ids'][0]: {tokenized_inputs['input_ids'][0]}\n"
        )

    return tokenized_inputs


print(f"\n##### Loading Model {args.model_name_or_path} ##### \n")

if "mistral" in args.model_name_or_path.lower():
    MODEL = MistralForTokenClassification
elif "llama" in args.model_name_or_path.lower():
    MODEL = LlamaForTokenClassification
else:
    raise NotImplementedError

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = MODEL.from_pretrained(
    args.model_name_or_path,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map={"": device_string},
    use_cache=False,
    cache_dir=None,
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules="all-linear",
    modules_to_save=None,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_true = [x for sublist in true_labels for x in sublist]
    flat_predicted = [x for sublist in true_predictions for x in sublist]

    label_names = [label for label in label2id.keys() if label not in ["ignore", "O"]]
    label_names.sort(key=lambda x: label2id[x])

    # Calculate sklearn classification report
    report = classification_report(
        flat_true, flat_predicted, labels=label_names, output_dict=True
    )

    weighted_avg = report["weighted avg"]

    return {
        "precision": weighted_avg["precision"],
        "recall": weighted_avg["recall"],
        "f1": weighted_avg["f1-score"],
    }


# Check if there is a checkpoint to resume from
if os.path.exists(output_dir):
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        latest_checkpoint = None
        print("No checkpoints found, starting training from scratch.")
else:
    latest_checkpoint = None
    print("No checkpoint directory found, starting training from scratch.")

print("Running Tokenization\n")

tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    remove_unused_columns=True,
    save_strategy="steps",
    save_steps=300,
    eval_strategy="steps",
    eval_steps=2000,
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    log_level="info",
    logging_steps=10,
    save_total_limit=2,
    report_to="wandb",
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False
)

wandb.init(
    project=f"{args.wandb_project}",
    name=f"Classifier_{save_mode}_{LANGUAGE}_seed{args.seed_number}",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Pass the latest checkpoint to trainer.train()
trainer.train(resume_from_checkpoint=latest_checkpoint)
trainer.save_model(final_model_dir)
