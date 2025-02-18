# inference.py
import argparse
from inference_models import (
    LlamaInference, Qwen2Inference, AyaInference,
    GemmaInference, MistralInference, PhiInference)
import pandas as pd
import json
import os
import jieba
import torch
import random
from tqdm import tqdm
import numpy as np  # Added for splitting data

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process language and model arguments.')
    parser.add_argument('--language', type=str, required=True, help='Language to process (e.g., "german" or "chinese")')
    parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., "llama", "qwen", "aya")')
    parser.add_argument('--mode', type=str, default=None, help='Leave empty if not "translated"')
    args = parser.parse_args()

    cache_dir = "/p/project/westai0015/code_julich"

    # Model paths for each model
    model_path_dict = {
        'llama-3.1' : "meta-llama/Llama-3.1-8B-Instruct",
        'llama-3.1-small' : "meta-llama/Llama-3.2-3B-Instruct",
        'qwen': "Qwen/Qwen2.5-7B-Instruct",
        'qwen-small' : "Qwen/Qwen2.5-3B-Instruct",
        'aya': "CohereForAI/aya-23-8B",
        'phi-mini': "microsoft/Phi-3.5-mini-instruct",
        'phi': "microsoft/Phi-3-small-8k-instruct",
        'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
        'gemma' : "google/gemma-2-9b-it",
        'gemma-small' : "google/gemma-2-2b-it"
    }

    # Mapping model names to their corresponding classes
    model_class_dict = {
        'llama-3.1-small': LlamaInference,
        'llama-3.1':LlamaInference,
        'qwen': Qwen2Inference,
        'qwen-small': Qwen2Inference,
        'aya': AyaInference,
        'phi-mini': PhiInference, 
        'phi': PhiInference,            
        'mistral': MistralInference,    
        'gemma' : GemmaInference,
        'gemma-small' : GemmaInference
    }

    if args.mode:
        mode = f'_{args.mode}'
    else:
        mode = ''
    language = args.language.lower()
    model_name = args.model.lower()

    # Check if the model name is valid
    if model_name not in model_class_dict:
        raise ValueError(f"Model '{model_name}' is not supported. Please choose from: {list(model_class_dict.keys())}")

    # Get the model class and path
    print('Loading Model and Tokenizer')
    InferenceModelClass = model_class_dict[model_name]
    model_path = model_path_dict[model_name]

    # Language code mapping
    lang_code_dict = {
        'amharic': 'am',
        'arabic': 'ar',
        'armenian': 'hy',
        'bangla':'bn',
        'basque': 'eu',
        'cantonese': 'zh-yue',
        'catalan': 'ca',
        'chinese': 'zh',
        'czech': 'cs',
        'esperanto':'eo',
        'french' : 'fr',
        'finnish':'fi',
        'german': 'de',
        'korean':'ko',
        'hebrew': 'he',
        'hindi': 'hi',
        'hungarian': 'hu',
        'indonesian': 'id',
        'italian' : 'it',
        'japanese': 'ja',
        'latin': 'la',
        'lithuanian':'lt',
        'malay' : 'ms',
        'persian': 'fa',
        'polish': 'pl',
        'portuguese' : 'pt',
        'romanian': 'ro',
        'russian': 'ru',
        'spanish': 'es',
        'swahili': 'sw',
        'serbian' : 'sr',
        'sindhi' : 'sd',
        'tamil': 'ta',
        'turkish': 'tr',
        'urdu': 'ur',
        'vietnamese': 'vi'
    }

    # Get the language code
    lang_code = lang_code_dict.get(language)
    if lang_code is None:
        raise ValueError(f"Language '{language}' is not supported. Please choose from: {list(lang_code_dict.keys())}")

    # Paths
    test_file = f"{language}/chat_format_{lang_code}{mode}.json"
    try:
        original_csv_path = f'{language}/wiki_references_and_queries_{lang_code}.csv'
        # Read the original CSV file containing references and queries
        original_df = pd.read_csv(original_csv_path)
    except:
        original_csv_path = f'{language}/wiki_references_and_queries_filtered_{lang_code}.csv'
        # Read the original CSV file containing references and queries
        original_df = pd.read_csv(original_csv_path)

    # Initialize the inference model
    print('Loading Model and Tokenizer....')
    inference_model = InferenceModelClass(model_path, cache_dir)

    # Load test data
    print('Loading Data....')
    test_data = load_json(test_file)

    # Split the original DataFrame into 5 parts
    original_df_splits = np.array_split(original_df, 5)

    # Define the separator token and import tokenizers as needed
    if language in ['chinese', 'cantonese']:
        sep_token = " reservedspecialtoken "
        import jieba
    elif language == 'japanese':
        sep_token = " reservedspecialtoken "
        from janome.tokenizer import Tokenizer as JapaneseTokenizer
        japanese_tokenizer = JapaneseTokenizer()
    else:
        sep_token = " <|reserved_special_token_200|> "

    # Seeds for each split
    seeds = [42, 43, 45, 47, 49]

    # Build a mapping from query content to test_data entries
    query_to_test_data_entry = {td[1]['content']: td for td in test_data}

    # Run the inference on each split
    for split_index, seed in enumerate(seeds):
        print(f"Processing split {split_index+1} with seed {seed}")

        # Check if the output JSON file already exists
        output_json_path = f'{language}/eval_inputs_{lang_code}_{model_name}{mode}_{split_index+1}.json'
        if os.path.exists(output_json_path):
            print(f"Output file {output_json_path} already exists. Skipping this split.")
            continue  # Skip to the next split

        # Set random seed
        torch.manual_seed(seed)
        random.seed(seed)

        # Get the corresponding split of original_df
        split_df = original_df_splits[split_index]

        # Collect the queries from 'query_1' and 'query_2' columns
        query_1_list = split_df['query_1'].tolist()
        query_2_list = split_df['query_2'].tolist()

        # Combine and get unique queries
        all_queries = list(set(query_1_list + query_2_list))

        # Get the test_data entries for the queries in all_queries
        test_data_split = [query_to_test_data_entry[query] for query in all_queries if query in query_to_test_data_entry]

        # Initialize lists to store queries and responses
        query_list = []
        response_list = []

        # Iterate through the test_data_split
        for i in tqdm(range(len(test_data_split)), desc=f'Generating split {split_index+1}'):
            # Get the query
            query = test_data_split[i][1]['content']

            # Generate the response
            response = inference_model.generate_response(test_data_split[i])
            print(f'Question: {query}\nAnswer: {response}')

            # Append data to lists
            query_list.append(query)
            response_list.append(response)

        # Create a DataFrame with the columns 'query' and 'response'
        responses_df = pd.DataFrame({
            'query': query_list,
            'response': response_list
        })

        # Adjust output filename to include split_index+1
        output_responses_csv = f'{language}/{language}_responses_{model_name}{mode}_{split_index+1}.csv'
        # Save the responses DataFrame to CSV
        responses_df.to_csv(output_responses_csv, index=False)

        print(f"CSV file generated successfully: {output_responses_csv}")

        # Now integrate the additional processing

        # Read the generated responses CSV
        generated_df = pd.read_csv(output_responses_csv)

        # Create a mapping from query to response
        query_to_response = dict(zip(generated_df['query'], generated_df['response']))

        # Map the responses to the split dataframe
        split_df['query_1_response'] = split_df['query_1'].map(query_to_response)
        split_df['query_2_response'] = split_df['query_2'].map(query_to_response)

        # Save the split dataframe with responses
        output_csv_path = f'{language}/{mode}final_dataset_with_responses_{model_name}_{split_index+1}.csv'
        split_df.to_csv(output_csv_path, index=False)

        print(f"Final dataset with responses saved to: {output_csv_path}")

        # Now process to create inputs and tokenize

        # Create a list to store the final results
        data_list = []

        # Iterate over each row in the split_df
        combine_col = ['reference_1', 'reference_2', 'reference_3', 'reference_4', 'reference_5']
        for index, row in split_df.iterrows():
            # Create the first input: references + separator token + query_1_response
            if language in ['german', 'chinese']:
                references = row['references']
            else:
                references = ' ### '.join([str(row[col]) for col in combine_col])
                
            input_1 = f"{references} {sep_token} {row['query_1_response']}"

            # Tokenize the input based on the language
            if language in ['chinese', 'cantonese']:
                input_1_tokens = list(jieba.cut(input_1))
                input_1_tokens = ['<|reserved_special_token_200|>' if token.strip() == 'reservedspecialtoken' else token for token in input_1_tokens]
            elif language == 'japanese':
                input_1_tokens = [token.surface for token in japanese_tokenizer.tokenize(input_1)]
                input_1_tokens = ['<|reserved_special_token_200|>' if token.strip() == 'reservedspecialtoken' else token for token in input_1_tokens]
            else:
                input_1_tokens = input_1.split()

            # Append to the list as a dictionary
            data_list.append({
                'input': input_1,
                'input_tokens': input_1_tokens
            })

            # Create the second input: references + separator token + query_2_response
            input_2 = f"{references} {sep_token} {row['query_2_response']}"

            # Tokenize the input
            if language in ['chinese', 'cantonese']:
                input_2_tokens = list(jieba.cut(input_2))
                input_2_tokens = ['<|reserved_special_token_200|>' if token.strip() == 'reservedspecialtoken' else token for token in input_2_tokens]
            elif language == 'japanese':
                input_2_tokens = [token.surface for token in japanese_tokenizer.tokenize(input_2)]
                input_2_tokens = ['<|reserved_special_token_200|>' if token.strip() == 'reservedspecialtoken' else token for token in input_2_tokens]
            else:
                input_2_tokens = input_2.split()

            # Append to the list as a dictionary
            data_list.append({
                'input': input_2,
                'input_tokens': input_2_tokens
            })

        # Save the data_list to a JSON file
        # output_json_path is already defined earlier
        with open(output_json_path, 'w') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        print(f"Inputs and tokens saved to JSON file: {output_json_path}")
