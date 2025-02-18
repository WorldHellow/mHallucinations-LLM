import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import json
import os
from billm import LlamaForTokenClassification, MistralForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report

def main(args):
    print(f"\nProcessing seed {args.seed}")
    process_seed(args, args.seed)

def process_seed(args, seed):
    # Set environment variables
    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_CACHE"] = args.transformers_cache

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()

    # Label mappings
    if 'Binary' in args.save_mode or 'binary' in args.save_mode:
        label2id = {"O": 0, "I-hallucination": 1, "ignore": -100}
    else:
        label2id = {
            'I-entity': 1, 'I-relation': 2,
            'I-invented': 3, 'I-contradictory': 4,
            'I-unverifiable': 5, 'I-subjective': 6,
            'O': 0, 'ignore': -100
        }
    id2label = {v: k for k, v in label2id.items()}
    label_list = [label for label in label2id.keys() if label not in ['ignore', 'O']]
    print(f"Label2ID: {label2id}")
    print(f"ID2Label: {id2label}")
    print(f"Label List: {label_list}")

    # Set variables
    SET = args.set_name
    lang = args.language.lower()
    save_mode = args.save_mode
    if 'Binary' in args.save_mode or 'binary' in args.save_mode:
        mode = 'bio_tags_binary'
    else:
        mode = 'tags'
    print(f'Mode = {mode}')

    # Create directories if they don't exist
    output_dir = os.path.join(args.output_base_dir, lang, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # **Added Functionality: Check if metrics already exist and are non-zero**
    metrics_path = os.path.join(output_dir, f'evaluation_metrics_{save_mode}.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            try:
                metrics = json.load(f)
                precision = metrics.get('Precision', 0)
                recall = metrics.get('Recall', 0)
                f1_score = metrics.get('F1 Score', 0)
                if precision != 0 and recall != 0 and f1_score != 0:
                    print(f"Metrics already computed and non-zero in {metrics_path}, skipping...")
                    return
            except (ValueError, KeyError) as e:
                print(f"Error reading metrics from {metrics_path}: {e}")
                pass  # Continue processing if there's an error

    # Load dataset
    data = load_json_dataset(args.data_file, lang, label2id, mode, args.max_samples)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    if 'mistral' in args.model_name.lower():
        tokenizer.add_special_tokens({'pad_token': '<unk>'})
    if 'llama' in args.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    checkpoint = args.checkpoint
    peft_config = PeftConfig.from_pretrained(checkpoint)

    if 'mistral' in args.model_name.lower():
        MODEL = MistralForTokenClassification
    elif 'llama' in args.model_name.lower():
        MODEL = LlamaForTokenClassification
    else:
        raise NotImplementedError

    model = MODEL.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        cache_dir=args.cache_dir
    )
    model = PeftModel.from_pretrained(model, checkpoint).to(args.device)
    model.eval()
    # Convert data to Dataset
    dataset = Dataset.from_dict(data)

    # Tokenize and align labels
    tokenized_inputs = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
        batched=True,
        remove_columns=['tokens', 'tags']
    )

    # Initialize Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args_training = TrainingArguments(
        output_dir='temp_output',
        per_device_eval_batch_size=args.batch_size,
        logging_dir='logs',
        report_to='none'
    )
    trainer = Trainer(model, args_training, data_collator=data_collator)

    # Make predictions
    raw_predictions, labels, _ = trainer.predict(tokenized_inputs)
    predictions = np.argmax(raw_predictions, axis=-1)

    # Process predictions and labels
    predicted_labels = [
        [int(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [int(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Merge subwords into words
    words, preds, trues, langs = merge_subwords(tokenizer, tokenized_inputs, predicted_labels, true_labels)

    # Convert IDs to labels
    predictions_read = [convert_ids_to_labels(pred, id2label) for pred in preds]
    labels_read = [convert_ids_to_labels(label, id2label) for label in trues]

    # Flatten lists
    flat_true = [x for sublist in labels_read for x in sublist]
    flat_predicted = [x for sublist in predictions_read for x in sublist]

    # Filter out 'O' and 'ignore' labels if necessary
    filtered_labels = [label for label in label_list if label not in ['ignore', 'O']]

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        flat_true,
        flat_predicted,
        labels=filtered_labels,
        average='macro',
        zero_division=0
    )

    # Compute classification report
    report = classification_report(
        flat_true,
        flat_predicted,
        labels=filtered_labels,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    ### **Save per-seed results**

    # Save metrics to JSON file (per-seed results)
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save the classification report to a text file
    report_path = os.path.join(output_dir, f'evaluation_report_{save_mode}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    ### **Optionally, save detailed predictions**

    # Save detailed predictions and labels to a JSON file (optional)
    detailed_results = {
        'words': words,
        'predictions': predictions_read,
        'true_labels': labels_read,
        'languages': langs
    }
    detailed_results_path = os.path.join(output_dir, f'detailed_results_{save_mode}.json')
    with open(detailed_results_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)

def load_json_dataset(file_path, language, label2id, mode, max_samples):
    def filter_language(data, lang, max_samples_per_lang):
        filtered_data = {'tokens': [], 'tags': [], 'language': []}
        count = 0
        for item in data:
            if item['language'].lower() == lang:
                if len(item['tokens']) != len(item[mode]):
                    print(f"Skipping due to length mismatch: tokens({len(item['tokens'])}) != tags({len(item[mode])})")
                    continue
                filtered_data['tokens'].append(item['tokens'])
                filtered_data['tags'].append([label2id.get(label, label2id['O']) for label in item[mode]])
                filtered_data['language'].append(item['language'])
                count += 1
                if max_samples_per_lang and count >= max_samples_per_lang:
                    break
            elif lang == 'all':
                # Include all languages if lang is 'all'
                if len(item['tokens']) != len(item[mode]):
                    print(f"Skipping due to length mismatch: tokens({len(item['tokens'])}) != tags({len(item[mode])})")
                    continue
                filtered_data['tokens'].append(item['tokens'])
                filtered_data['tags'].append([label2id.get(label, label2id['O']) for label in item[mode]])
                filtered_data['language'].append(item['language'])
                count += 1
                if max_samples_per_lang and count >= max_samples_per_lang:
                    break
        return filtered_data

    with open(file_path, 'r') as f:
        data = json.load(f)

    data_dict = filter_language(data, language, max_samples)
    print(f"Number of samples: {len(data_dict['tokens'])}")
    return data_dict

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        max_length=4096,
        truncation=True
    )

    labels = []
    word_ids = []
    label_all_tokens = True
    special_token_id = 128205  # ID for |reserved_special_token_200|

    for i, label in enumerate(examples["tags"]):
        word_id = tokenized_inputs.word_ids(batch_index=i)
        word_ids.append(word_id)
        previous_word_idx = None
        label_ids = []
        special_token_found = False

        for idx, word_idx in enumerate(word_id):
            if tokenized_inputs.input_ids[i][idx] == special_token_id:
                special_token_found = True

            if not special_token_found:
                label_ids.append(-100)
            else:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids
    tokenized_inputs["language"] = examples["language"]  # Preserve 'language'

    return tokenized_inputs

def merge_subwords(tokenizer, tokenized_inputs, predicted_labels, true_labels):
    words = []
    preds = []
    trues = []
    langs = []

    # Special token ID for |reserved_special_token_200|
    special_token_id = 128205

    # Iterate through the tokenized inputs
    for i, input_id in tqdm(enumerate(tokenized_inputs["input_ids"]), desc='Merging Subwords', total=len(tokenized_inputs["input_ids"])):
        word_id = tokenized_inputs["word_ids"][i]
        current_words = []
        current_preds = []
        current_trues = []
        word_tokens = []

        try:
            # Find the index of the special token
            special_token_index = input_id.index(special_token_id)
        except ValueError:
            # Skip this entry if the special token is not found
            continue

        current_word_idx = None
        for idx, word_idx in enumerate(word_id[special_token_index:], start=special_token_index):
            if word_idx is not None:
                if word_idx != current_word_idx:
                    current_word_idx = word_idx
                    if word_tokens:
                        current_words.append(tokenizer.decode(word_tokens))
                        word_tokens = []
                        word_pos = len(current_words) - 1
                        try:
                            current_preds.append(predicted_labels[i][word_pos])
                            current_trues.append(true_labels[i][word_pos])
                        except IndexError:
                            print(f"IndexError: i={i}, idx={idx}, len(current_words)={len(current_words)}, len(predicted_labels[i])={len(predicted_labels[i])}, len(true_labels[i])={len(true_labels[i])}")
                            print(f"Predicted labels: {predicted_labels[i]}")
                            print(f"True labels: {true_labels[i]}")
                            raise
                word_tokens.append(input_id[idx])

        if word_tokens:
            current_words.append(tokenizer.decode(word_tokens))
            word_pos = len(current_words) - 1
            try:
                current_preds.append(predicted_labels[i][word_pos])
                current_trues.append(true_labels[i][word_pos])
            except IndexError:
                print(f"IndexError at end: i={i}, len(current_words)={len(current_words)}, len(predicted_labels[i])={len(predicted_labels[i])}, len(true_labels[i])={len(true_labels[i])}")
                print(f"Predicted labels: {predicted_labels[i]}")
                print(f"True labels: {true_labels[i]}")
                raise

        words.append(current_words)
        preds.append(current_preds)
        trues.append(current_trues)
        langs.append(tokenized_inputs["language"][i])  # Now 'language' exists

    return words, preds, trues, langs

def convert_ids_to_labels(id_list, id2label):
    return [id2label.get(id, 'O') for id in id_list]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on Test Set")
    parser.add_argument("--hf_home", type=str, default=None, help="HF home directory")
    parser.add_argument("--transformers_cache", type=str, default=None, help="Transformers cache directory")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--set_name", type=str, default="test", help="Dataset set name (e.g., 'test')")
    parser.add_argument("--language", type=str, required=True, help="Language")
    parser.add_argument("--save_mode", type=str, required=True, help="Save mode name (model name)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load")
    parser.add_argument("--output_base_dir", type=str, default="test_set_results", help="Base directory for output files")
    args = parser.parse_args()
    main(args)
