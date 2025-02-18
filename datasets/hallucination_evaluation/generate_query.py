#!/usr/bin/env python3

import pandas as pd
import openai
from tqdm import tqdm
import json
import os
from datetime import datetime
import tiktoken
import argparse
from langdetect import detect, LangDetectException
import random

# -----------------------------
# Configuration and Constants
# -----------------------------
WHITE_SPACE_LANG = ['chinese', 'japanese', 'cantonese']
API_KEY_PATH = ...  # Update if necessary
WIKI_LANGS_CSV = 'wiki_langs.csv'  # Path to the wiki_langs.csv file

# Load OpenAI API Key
try:
    with open(API_KEY_PATH, 'r') as file:
        key = json.load(file)
    openai.api_key = key
except FileNotFoundError:
    raise FileNotFoundError(f"API key file not found at {API_KEY_PATH}. Please provide a valid path.")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the 'saad_open_ai_key' in the JSON file.")

# **Define Pricing Rates**
# Update these rates based on OpenAI's official pricing. These are illustrative examples.
PRICING = {
    'gpt-4o': {'prompt': 0.0025, 'completion': 0.01}
}

# **Log File Path**
LOG_FILE_PATH = 'wiki-articles-all/api_calls_log.jsonl'  # JSON Lines format for structured logging

# **Billing CSV Path**
TOTAL_BILLING_CSV = 'total_billing_queries.csv'

# **Base Paths**
BASE_ARTICLES_PATH = 'wiki-articles-all'
# BASE_QUERIES_PATH = 'wiki-queries'  # Removed as output will now be within wiki-articles-all

# -----------------------------
# Helper Functions
# -----------------------------

def log_api_call(log_entry, log_file=LOG_FILE_PATH):
    """
    Logs the API call details and cost to a JSON Lines file.

    Args:
        log_entry (dict): The log entry containing API call details.
        log_file (str): Path to the log file.
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"API call logged: ${log_entry['cost_usd']:.6f}")
    except Exception as e:
        print(f"Failed to log API call: {e}")

def compute_cost(model, prompt_tokens, completion_tokens):
    """
    Computes the cost of an API call based on token usage and pricing.

    Args:
        model (str): The model used for the API call.
        prompt_tokens (int): Number of prompt tokens.
        completion_tokens (int): Number of completion tokens.

    Returns:
        float: The total cost in USD.
    """
    model_pricing = PRICING.get(model)
    if not model_pricing:
        raise ValueError(f"Pricing for model '{model}' is not defined.")

    prompt_cost = (prompt_tokens / 1000) * model_pricing['prompt']
    completion_cost = (completion_tokens / 1000) * model_pricing['completion']
    total_cost = prompt_cost + completion_cost
    return round(total_cost, 6)

def count_tokens(text, model='gpt-4o'):
    """
    Counts the number of tokens in a given text using the appropriate tokenizer.

    Args:
        text (str): The text to tokenize.
        model (str): The model name to determine the tokenizer.

    Returns:
        int: The number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to a default encoding if the model is not found
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def is_english(text):
    """
    Detects if the given text is in English.

    Args:
        text (str): The text to detect.

    Returns:
        bool: True if the text is in English, False otherwise.
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def log_and_print(message):
    """
    Logs a message to the console.

    Args:
        message (str): The message to print.
    """
    print(message)

def get_language_code(language):
    """
    Retrieves the language code for the given language from wiki_langs.csv.

    Args:
        language (str): The language name in capitalized form.

    Returns:
        str: The corresponding language code.

    Raises:
        ValueError: If the language is not found in the CSV.
    """
    if not os.path.exists(WIKI_LANGS_CSV):
        raise FileNotFoundError(f"{WIKI_LANGS_CSV} not found. Please ensure the file exists.")

    df_langs = pd.read_csv(WIKI_LANGS_CSV)
    df_lang = df_langs[df_langs['Language'].str.lower() == language.lower()]

    if df_lang.empty:
        raise ValueError(f"Language '{language}' not found in {WIKI_LANGS_CSV}.")

    language_code = df_lang.iloc[0]['Language Code']
    return language_code

# -----------------------------
# Core Functionality
# -----------------------------

def generate_queries(references, lang, model, billing_accumulator):
    """
    Generate two knowledge-intensive queries based on the given references using the chat completion API.
    Also computes and logs the cost of the API call.

    Args:
        references (list): List of reference strings.
        lang (str): Language name.
        model (str): Model to use for generation.
        billing_accumulator (dict): Dictionary to accumulate billing information.

    Returns:
        tuple: A tuple containing two generated queries.
    """
    # Adjust the language name for certain cases
    if lang.lower() == 'chinese':
        lang_name = 'Mandarin'
    else:
        lang_name = lang

    # Construct the prompt
    prompt = (
        "Given the following references:\n" +
        "\n".join([f"Reference {i+1}: {ref}" for i, ref in enumerate(references) if ref]) +
        f"\nGenerate TWO knowledge-intensive queries in {lang_name}. "
        "Make sure the questions are brief but knowledge-intensive. "
        "The questions should be such that the answer requires thorough reading of the reference text. "
        "Separate them with a new line."
    )

    try:
        # Count prompt tokens before making the API call (optional but useful)
        prompt_token_count = count_tokens(prompt, model=model)

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant in {lang_name} language."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Extract token usage from the response
        usage = response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', prompt_token_count)  # Fallback to pre-counted tokens
        completion_tokens = usage.get('completion_tokens', 0)

        # Compute cost
        cost = compute_cost(model, prompt_tokens, completion_tokens)

        # Update billing accumulator
        billing_accumulator['total_input_prompt_cost'] += (prompt_tokens / 1000) * PRICING[model]['prompt']
        billing_accumulator['total_completion_response_cost'] += (completion_tokens / 1000) * PRICING[model]['completion']
        billing_accumulator['total_generated_tokens'] += prompt_tokens + completion_tokens
        billing_accumulator['total_cost'] += cost

        # Prepare log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'response_id': response.get('id', 'unknown'),
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cost_usd': cost
        }

        # Log the API call
        log_api_call(log_entry)

        # Extract queries from the response
        queries = response['choices'][0]['message']['content'].strip().split('\n')
        # Ensure exactly two queries
        if len(queries) < 2:
            queries.append("")
        return queries[:2]  # Return exactly two queries

    except Exception as e:
        log_and_print(f"Error generating queries: {e}")
        # Log the failed API call with zero cost or appropriate handling
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'response_id': 'error',
            'model': model,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cost_usd': 0.0,
            'error': str(e)
        }
        log_api_call(log_entry)
        return ["", ""]

def process_dataframe(df, lang, model, billing_accumulator):
    """
    Process the DataFrame to generate queries for each row.
    Skip rows where all references are empty or if there is only one reference.

    Args:
        df (pd.DataFrame): The input DataFrame containing references.
        lang (str): Language name.
        model (str): Model to use for generation.
        billing_accumulator (dict): Dictionary to accumulate billing information.

    Returns:
        pd.DataFrame: DataFrame with added query columns.
    """
    queries = {'query_1': [], 'query_2': []}
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating Queries for {lang}"):
        # Collect references and filter out empty ones
        references = [
            row['reference_1'] if isinstance(row['reference_1'], str) else '',
            row['reference_2'] if isinstance(row['reference_2'], str) else '',
            row['reference_3'] if isinstance(row['reference_3'], str) else '',
            row['reference_4'] if isinstance(row['reference_4'], str) else '',
            row['reference_5'] if isinstance(row['reference_5'], str) else '',
        ]

        # Filter non-empty references
        references = [ref for ref in references if ref.strip()]  # Keep only non-empty references

        # Skip the row if all references are empty or if there is only one reference
        if lang.lower() not in WHITE_SPACE_LANG:
            if len(references) <= 1:
                queries['query_1'].append('')  # Placeholder for skipped row
                queries['query_2'].append('')  # Placeholder for skipped row
                continue

        # Generate queries based on the valid references
        query_1, query_2 = generate_queries(references, lang, model=model, billing_accumulator=billing_accumulator)

        # Append queries to the respective lists
        queries['query_1'].append(query_1)
        queries['query_2'].append(query_2)

    # Add the queries back to the DataFrame
    df['query_1'] = queries['query_1']
    df['query_2'] = queries['query_2']

    return df

def write_billing_summary(language, billing_accumulator, output_folder):
    """
    Writes the billing summary to a text file for the given language.

    Args:
        language (str): The language name.
        billing_accumulator (dict): The billing information for the language.
        output_folder (str): The path to the language-specific output folder.
    """
    filename = os.path.join(output_folder, f"bill_{language}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Billing Summary for {language}\n")
        f.write(f"{'-'*40}\n")
        f.write(f"Model Used: {billing_accumulator['model_used']}\n")
        f.write(f"Cost per 1000 Tokens for Input/Prompt: ${billing_accumulator['cost_per_1000_prompt']:.6f}\n")
        f.write(f"Cost per 1000 Tokens for Completion: ${billing_accumulator['cost_per_1000_completion']:.6f}\n")
        f.write(f"Total Input/Prompt Cost: ${billing_accumulator['total_input_prompt_cost']:.6f}\n")
        f.write(f"Total Completion/Response Cost: ${billing_accumulator['total_completion_response_cost']:.6f}\n")
        f.write(f"Total Generated Tokens for {language}: {billing_accumulator['total_generated_tokens']}\n")
        f.write(f"Total Cost: ${billing_accumulator['total_cost']:.6f}\n")

def update_total_billing_csv(billing_info_list):
    """
    Creates or updates the total_billing_queries.csv file with the provided billing information.

    Args:
        billing_info_list (list): List of billing information dictionaries for each language.
    """
    # Define the columns for the CSV
    columns = [
        'language',
        'model_used',
        'cost_per_1000_prompt',
        'cost_per_1000_completion',
        'total_input_prompt_cost',
        'total_completion_response_cost',
        'total_generated_tokens',
        'total_cost'
    ]

    # Convert the billing_info_list to a DataFrame
    new_billing_df = pd.DataFrame(billing_info_list, columns=columns)

    if os.path.exists(TOTAL_BILLING_CSV):
        # Load existing CSV
        existing_df = pd.read_csv(TOTAL_BILLING_CSV)
        # Merge with new billing info
        merged_df = pd.concat([existing_df, new_billing_df], ignore_index=True)
        # Optionally, remove duplicates based on 'language' by keeping the latest entry
        merged_df.drop_duplicates(subset=['language'], keep='last', inplace=True)
    else:
        merged_df = new_billing_df

    # Save back to CSV
    merged_df.to_csv(TOTAL_BILLING_CSV, index=False)
    print(f"Total billing information updated in {TOTAL_BILLING_CSV}")

def filter_queries(input_file_path):
    """
    Filters out English queries and removes rows where both queries are empty.

    Args:
        input_file_path (str): Path to the input JSONL file containing generated queries.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Load the JSON lines file
    df = pd.read_json(input_file_path, lines=True)

    # Filter out rows where either query_1 or query_2 is in English
    df_filtered = df[~df['query_1'].apply(is_english) & ~df['query_2'].apply(is_english)]

    # **Remove rows where both query_1 and query_2 are empty or NaN**
    df_filtered = df_filtered[
        ~(
            df_filtered['query_1'].fillna('').str.strip().eq('') & 
            df_filtered['query_2'].fillna('').str.strip().eq('')
        )
    ]

    return df_filtered

def save_filtered_csv(df_filtered, output_folder, lang_code):
    """
    Saves the filtered DataFrame to a CSV file.

    Args:
        df_filtered (pd.DataFrame): The filtered DataFrame.
        output_folder (str): Path to the language-specific output folder.
        lang_code (str): Language code for naming the output file.
    """
    filtered_file_name = f"wiki_references_and_queries_filtered_{lang_code}.csv"
    filtered_file_path = os.path.join(output_folder, filtered_file_name)
    df_filtered.to_csv(filtered_file_path, index=False)
    print(f"Filtered queries saved to {filtered_file_path}")

def convert_to_chat_format(df_filtered, language, output_folder, lang_code):
    """
    Converts the filtered queries into chat format JSON.

    Args:
        df_filtered (pd.DataFrame): The filtered DataFrame.
        language (str): Language name.
        output_folder (str): Path to the language-specific output folder.
        lang_code (str): Language code for naming the output file.
    """
    # Initialize an empty list to store chat format data
    chat_format_data = []

    # Iterate over the filtered rows of the DataFrame
    for _, row in df_filtered.iterrows():
        query_1 = row['query_1']
        query_2 = row['query_2']

        # Check if query_1 is non-empty and not NaN
        if isinstance(query_1, str) and query_1.strip():
            # Add the first instance with query_1
            chat_format_data.append([
                {
                    "role": "system",
                    "content": f"You are a helpful {language} AI assistant. Your answer must be in {language}."
                },
                {
                    "role": "user",
                    "content": query_1
                }
            ])

        # Check if query_2 is non-empty and not NaN
        if isinstance(query_2, str) and query_2.strip():
            # Add the second instance with query_2
            chat_format_data.append([
                {
                    "role": "system",
                    "content": f"You are a helpful {language} AI assistant. Your answer must be in {language}."
                },
                {
                    "role": "user",
                    "content": query_2
                }
            ])

    # Define the output file path
    output_file_name = f"chat_format_{lang_code}.json"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Write the chat format data to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(chat_format_data, json_file, ensure_ascii=False, indent=4)

    print(f"Chat format data saved to {output_file_path}\n")

# -----------------------------
# Main Execution
# -----------------------------

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Generate and process queries for a specific language.")
    parser.add_argument('language', type=str, help='Name of the language (e.g., Japanese)')
    args = parser.parse_args()

    language = args.language

    try:
        # Retrieve language code from wiki_langs.csv
        lang_code = get_language_code(language)
    except Exception as e:
        log_and_print(f"Error retrieving language code: {e}")
        return

    # Create language-specific output folder in lowercase within wiki-articles-all
    language_folder = language.lower()
    output_path = os.path.join(BASE_ARTICLES_PATH, language_folder)
    os.makedirs(output_path, exist_ok=True)

    # Define input and output file paths
    input_references_file = f"wiki_references_{lang_code}.csv"
    input_file_path = os.path.join(BASE_ARTICLES_PATH, language_folder, input_references_file)
    if not os.path.exists(input_file_path):
        log_and_print(f"Input file for {language} not found: {input_file_path}")
        return

    # Load the references DataFrame
    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        log_and_print(f"Error reading input CSV file: {e}")
        return

    log_and_print(f"Processing {len(df)} articles for {language} ({lang_code})")

    # Initialize billing accumulator for the language
    billing_accumulator = {
        'model_used': 'gpt-4o',  # Assuming 'gpt-4o' as default, adjust if needed
        'cost_per_1000_prompt': PRICING['gpt-4o']['prompt'],
        'cost_per_1000_completion': PRICING['gpt-4o']['completion'],
        'total_input_prompt_cost': 0.0,
        'total_completion_response_cost': 0.0,
        'total_generated_tokens': 0,
        'total_cost': 0.0
    }

    # Generate queries and append to DataFrame
    df_with_queries = process_dataframe(df, language, model='gpt-4o', billing_accumulator=billing_accumulator)

    # Save the DataFrame with queries to JSON Lines format in the language-specific folder
    queries_output_file = f"wiki_references_queries_{lang_code}.jsonl"
    queries_output_path = os.path.join(output_path, queries_output_file)
    try:
        df_with_queries.to_json(queries_output_path, orient='records', lines=True)
        log_and_print(f"Generated queries saved to {queries_output_path}\n")
    except Exception as e:
        log_and_print(f"Failed to save generated queries to {queries_output_path}: {e}")
        return

    # Write billing summary to bill_language.txt in the language-specific folder
    write_billing_summary(language, billing_accumulator, output_path)
    log_and_print(f"Billing summary written to bill_{language}.txt\n")

    # Prepare billing info for total_billing_queries.csv
    billing_info = {
        'language': language,
        'model_used': billing_accumulator['model_used'],
        'cost_per_1000_prompt': billing_accumulator['cost_per_1000_prompt'],
        'cost_per_1000_completion': billing_accumulator['cost_per_1000_completion'],
        'total_input_prompt_cost': round(billing_accumulator['total_input_prompt_cost'], 6),
        'total_completion_response_cost': round(billing_accumulator['total_completion_response_cost'], 6),
        'total_generated_tokens': billing_accumulator['total_generated_tokens'],
        'total_cost': round(billing_accumulator['total_cost'], 6)
    }

    # Update total_billing_queries.csv with the new billing info
    update_total_billing_csv([billing_info])

    # -----------------------------
    # Additional Functionality
    # -----------------------------

    # Filter generated queries and convert to chat format
    log_and_print(f"Filtering generated queries for {language} ({lang_code})")

    # Path to the generated queries JSONL
    generated_queries_path = queries_output_path

    # Check if the generated queries file exists
    if not os.path.exists(generated_queries_path):
        log_and_print(f"Generated queries file not found: {generated_queries_path}. Skipping filtering and chat format conversion.")
        return

    # Filter queries
    try:
        df_filtered = filter_queries(generated_queries_path)
        log_and_print(f"Filtered {len(df_filtered)} queries after removing English and empty entries.")
    except Exception as e:
        log_and_print(f"Error filtering queries: {e}")
        return

    # Save the filtered DataFrame to a new CSV file in the language-specific folder
    save_filtered_csv(df_filtered, output_path, lang_code)

    # Convert filtered queries to chat format JSON
    convert_to_chat_format(df_filtered, language, output_path, lang_code)

if __name__ == "__main__":
    main()
