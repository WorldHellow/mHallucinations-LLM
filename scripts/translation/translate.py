import json
from nltk.tokenize import sent_tokenize
import argparse
from tqdm import tqdm
import os
import ctranslate2
import sentencepiece as spm

# Define tag replacements with unique identifiers and special characters
TAG_MAP = {
    '<subjective>': '__TAG_SUBJECTIVE_START__',
    '</subjective>': '__TAG_SUBJECTIVE_END__',
    '<invented>': '__TAG_INVENTED_START__',
    '</invented>': '__TAG_INVENTED_END__',
    '<unverifiable>': '__TAG_UNVERIFIABLE_START__',
    '</unverifiable>': '__TAG_UNVERIFIABLE_END__',
    '<entity>': '__TAG_ENTITY_START__',
    '</entity>': '__TAG_ENTITY_END__',
    '<relation>': '__TAG_RELATION_START__',
    '</relation>': '__TAG_RELATION_END__',
    '<contradictory>': '__TAG_CONTRADICTORY_START__',
    '</contradictory>': '__TAG_CONTRADICTORY_END__',
    '<delete>': '__TAG_DELETE_START__',
    '</delete>': '__TAG_DELETE_END__',
    '<mark>': '__TAG_MARK_START__',
    '</mark>': '__TAG_MARK_END__'
}

def replace_tags_with_delimiters(text):
    for tag, delimiter in TAG_MAP.items():
        text = text.replace(tag, delimiter)
    return text

def revert_delimiters_to_tags(text):
    for tag, delimiter in TAG_MAP.items():
        text = text.replace(delimiter, tag)
    return text

def tokenize_paragraphs(text):
    return sent_tokenize(text)

def run_inference_batch(translator, sp_processor, sentences, src_lang, tgt_lang, batch_size=128):
    translated_texts = []
    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start+batch_size]
        source_sentences = [sent.strip() for sent in batch]
        source_sents_subworded = sp_processor.encode_as_pieces(source_sentences)
        source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]
        target_prefix = [[tgt_lang]] * len(source_sents_subworded)
        
        translations_subworded = translator.translate_batch(
            source_sents_subworded, 
            batch_type="tokens", 
            max_batch_size=2024, 
            beam_size=4, 
            target_prefix=target_prefix
        )
        translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
        for translation in translations_subworded:
            if tgt_lang in translation:
                translation.remove(tgt_lang)
        translations = sp_processor.decode(translations_subworded)
        translated_texts.extend(translations)
    
    return ' '.join(translated_texts)

def translate_content(content, translator, sp_processor, src_lang, tgt_lang):
    content_with_delimiters = replace_tags_with_delimiters(content)
    sentences = tokenize_paragraphs(content_with_delimiters)
    translated_text = run_inference_batch(translator, sp_processor, sentences, src_lang, tgt_lang)
    return revert_delimiters_to_tags(translated_text)

def translate_json_data(data, translator, sp_processor, src_lang, tgt_lang, temp_file_path):
    user_contents = []
    assistant_contents = []
    roles = []

    for item in tqdm(data, desc="Extracting messages"):
        if 'messages' in item:
            for message in item['messages']:
                role = message['role']
                content = message['content']
                roles.append(role)
                if role == 'user':
                    user_contents.append(content)
                elif role == 'assistant':
                    assistant_contents.append(content)

    translated_user_contents = []
    translated_assistant_contents = []

    print("Translating user messages...")
    for idx, content in enumerate(tqdm(user_contents)):
        translated_user_contents.append(translate_content(content, translator, sp_processor, src_lang, tgt_lang))
        if (idx + 1) % 50 == 0:
            save_intermediate_results(temp_file_path, translated_user_contents, translated_assistant_contents, 'user')

    print("Translating assistant messages...")
    for idx, content in enumerate(tqdm(assistant_contents)):
        translated_assistant_contents.append(translate_content(content, translator, sp_processor, src_lang, tgt_lang))
        if (idx + 1) % 50 == 0:
            save_intermediate_results(temp_file_path, translated_user_contents, translated_assistant_contents, 'assistant')

    save_intermediate_results(temp_file_path, translated_user_contents, translated_assistant_contents, 'final')

    translated_data = []
    user_idx = 0
    assistant_idx = 0

    for item in tqdm(data, desc="Reconstructing translated messages"):
        if 'messages' in item:
            translated_item = {'messages': []}
            for message in item['messages']:
                role = message['role']
                if role == 'user':
                    translated_content = translated_user_contents[user_idx]
                    translated_item['messages'].append({'role': role, 'content': translated_content})
                    user_idx += 1
                elif role == 'assistant':
                    translated_content = translated_assistant_contents[assistant_idx]
                    translated_item['messages'].append({'role': role, 'content': translated_content})
                    assistant_idx += 1
            translated_data.append(translated_item)

    return translated_data

def save_intermediate_results(temp_file_path, user_contents, assistant_contents, role):
    temp_data = {
        'user_contents': user_contents,
        'assistant_contents': assistant_contents
    }
    temp_role_file_path = f"{temp_file_path}_{role}.json"
    with open(temp_role_file_path, 'w', encoding='utf-8') as file:
        json.dump(temp_data, file, ensure_ascii=False, indent=4)
    print(f"Intermediate results saved to {temp_role_file_path}")

def process_files(file_list, src_lang, tgt_lang):
    # Load CTranslate2 model and SentencePiece processor
    print('Loading CTranslate2 model and SentencePiece processor.')
    ct_model_path = "nllb-200-1.3B-int8"
    sp_model_path = "flores200_sacrebleu_tokenizer_spm.model"

    translator = ctranslate2.Translator(ct_model_path, device="cuda")
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(sp_model_path)

    # Translate each file
    for file_path in file_list:
        print(f"\nTranslating file: {file_path}")
        temp_file_path = f'temp_{tgt_lang}_{os.path.basename(file_path)}'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if 'test' in file_path:
            data = data[:300]
        
        translated_data = translate_json_data(data, translator, sp_processor, src_lang, tgt_lang, temp_file_path)

        output_file_path = f'translated_{tgt_lang}_{file_path}'
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(translated_data, file, ensure_ascii=False, indent=4)

        print(f"Translation completed and saved to {output_file_path}.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="Choose language tag from German, Chinese, Turkish, Arabic, Russian.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    language_map = {
        'german': 'deu_Latn',
        'arabic': 'arb_Arab',
        'turkish': 'tur_Latn',
        'russian': 'rus_Cyrl',
        'chinese': 'zho_Hans'
    }

    if args.target_lang.lower() not in language_map:
        raise ValueError("Unsupported target language")

    language = language_map[args.target_lang.lower()]
    
    files_to_process = ['test_dataset_detection.json', 'train_dataset_detection.json']
    process_files(files_to_process, 'eng_Latn', language)
