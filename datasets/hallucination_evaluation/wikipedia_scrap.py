#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import os
import sys
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session():
    """
    Creates a requests Session with a retry mechanism.
    Retries on specific HTTP status codes and network-related errors.
    """
    session = requests.Session()
    retry = Retry(
        total=5,  # Total number of retries
        backoff_factor=0.3,  # A backoff factor to apply between attempts
        status_forcelist=(500, 502, 504),  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # HTTP methods to retry on
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_random_article(lang_code, session):
    """
    Fetch a random Wikipedia article in the given language using the provided session.
    Implements retries via the session's adapter.
    """
    URL = f"https://{lang_code}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,  # Only fetch articles (not talk pages, files, etc.)
        "rnlimit": 1       # Limit to one random article at a time
    }
    
    try:
        response = session.get(url=URL, params=PARAMS, timeout=10)
        response.raise_for_status()
        data = response.json()
        random_article = data['query']['random'][0]
        article_title = random_article['title']
        article_url = f"https://{lang_code}.wikipedia.org/wiki/{article_title.replace(' ', '_')}"
        return article_title, article_url
    except Exception as e:
        print(f"Error fetching random article for language code '{lang_code}': {e}")
        return None, None

def clean_text(html_content):
    """
    Clean HTML content and return the main text:
      - Remove <script>, <style>, <header>, <footer>, <nav>, <aside>, <form>
      - If a page has a 'noarticletext' div/table inside bodyContent, returns an empty string.
      - Finally, remove any leftover HTML tags, new lines, and extra spaces.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove elements that are not part of the main content
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        element.decompose()
    
    # Get the content of the article under <div id="bodyContent">
    body_content = soup.find('div', id='bodyContent')
    if not body_content:
        body_content = soup

    # Check for pages that have no article text
    no_article_div = body_content.find('div', class_="noarticletext mw-content-rtl")
    no_article_table = body_content.find('table', id="noarticletext")
    if no_article_div or no_article_table:
        # No article text present, return empty string
        return ""

    # Extract text from all paragraphs
    paragraphs = body_content.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    
    # Remove leftover HTML tags (in case of any inline tags) and newlines
    text = re.sub(r'<[^>]+>', '', text)  
    text = text.replace('\n', ' ')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_article_content(url, session):
    """
    Fetch the article content, clean it, and return the cleaned text.
    """
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        html_content = response.content
        text = clean_text(html_content)
        return text
    except Exception as e:
        print(f"Error fetching article content from '{url}': {e}")
        return ""

def get_article_categories(title, lang_code, session):
    """
    Fetch the top category of the Wikipedia article using the Wikipedia API.
    Works for any language and removes the 'Category:' prefix regardless of language.
    """
    URL = f"https://{lang_code}.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "categories",
        "titles": title,
        "format": "json",
        "cllimit": 1  # Fetch only the first category
    }
    
    try:
        response = session.get(url=URL, params=PARAMS, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        for page_id, page_data in pages.items():
            if 'categories' in page_data:
                # Get the first category
                first_category = page_data['categories'][0]['title']
                # Remove any 'Category:' prefix in the local language
                cleaned_category = first_category.split(":")[-1].strip()
                return cleaned_category
        return 'No Category'
    except Exception as e:
        print(f"Error fetching categories for article '{title}' in language code '{lang_code}': {e}")
        return 'No Category'

def load_existing_titles(csv_path):
    """
    Load existing article titles from the specified CSV file.
    Returns a set of titles. If the file doesn't exist, returns an empty set.
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return set(df['Article Title'].tolist())
        except Exception as e:
            print(f"Error reading '{csv_path}': {e}")
            return set()
    return set()

def scrape_wikipedia_articles(lang_name,
                              lang_code, 
                              max_articles=1000, 
                              max_attempts=5000, 
                              collected_titles=None):
    """
    Scrape random Wikipedia articles in a specific language.
    For Chinese/Cantonese/Japanese, require at least 1000 characters.
    For other languages, require at least 200 words.
    
    - Skips articles that have 'noarticletext'
    - Skips articles that fail threshold
    - Returns a list of dicts with:
        'Article Title', 'URL', 'Word Count', 'Character Count', 'Category', 'Content'
    """
    # Define language sets for which we use the 1000-char rule
    no_whitespace_langs = ["Chinese", "Cantonese", "Japanese"]

    articles = []
    attempts = 0
    failed_attempts = 0
    
    session = get_session()
    
    with tqdm(total=max_articles, desc=f"Scraping articles from {lang_code}") as pbar:
        while len(articles) < max_articles and attempts < max_attempts:
            title, url = get_random_article(lang_code, session)
            attempts += 1
            
            if not url:
                failed_attempts += 1
                continue  # Skip if fetching random article failed
            
            # Check for duplicates
            if title in collected_titles:
                # print(f"Duplicate article found: '{title}'. Skipping.")
                continue  # Skip duplicates
            
            # Get the cleaned body content
            text = get_article_content(url, session)
            if not text:
                # Either noarticletext or some fetch error
                continue

            # Compute character count and word count
            char_count = len(text)  # total characters in BODY content
            word_count = len(text.split())  # total words in BODY content
            
            # Check thresholds
            if lang_name in no_whitespace_langs:
                # Must be >= 1000 characters
                if char_count < 600:
                    continue
            else:
                # Must be >= 200 words
                if word_count < 200:
                    continue

            # If we reach here, article meets the thresholds
            categories = get_article_categories(title, lang_code, session)
            
            articles.append({
                'Article Title': title,
                'URL': url,
                'Word Count': word_count,
                'Character Count': char_count,
                'Category': categories,
                'Content': text
            })
            
            collected_titles.add(title)
            pbar.update(1)  # Update progress bar
            time.sleep(random.uniform(0.5, 1.5))  # random delay to reduce load
            
            if attempts % 100 == 0:
                print(f"Attempts: {attempts}, Valid: {len(articles)}, Failed: {failed_attempts}")

        if len(articles) < max_articles:
            print(f"Reached maximum attempts ({max_attempts}). "
                  f"Collected {len(articles)} articles for language '{lang_code}'.")
        else:
            print(f"Successfully collected {len(articles)} articles for language '{lang_code}'.")
    
    return articles

def split_into_references(text, max_references=5, max_sentences=3):
    """
    Split text into references, each with a maximum of max_sentences sentences.
    """
    sentence_endings = r'(?<=[.!?۔！？])\s+'
    sentences = re.split(sentence_endings, text)
    references = []
    
    current_reference = []
    for sentence in sentences:
        if len(current_reference) >= max_sentences:
            references.append(" ".join(current_reference).strip())
            current_reference = []
        current_reference.append(sentence)
        if len(references) >= max_references:
            break
    
    if current_reference and len(references) < max_references:
        references.append(" ".join(current_reference).strip())
    
    # Ensure exactly max_references references, padding if needed
    while len(references) < max_references:
        references.append("")
    
    return references

def build_dataset_with_references(articles):
    """
    Build a dataset with articles' references.
    """
    reference_columns = ['reference_1', 'reference_2', 'reference_3', 'reference_4', 'reference_5']
    new_data = []
    
    for article in tqdm(articles, desc="Processing Articles"):
        text = article['Content']
        references = split_into_references(text)
        new_data.append({
            'Category': article['Category'],
            'Article Title': article['Article Title'],
            'URL': article['URL'],
            'Word Count': article['Word Count'],
            'Character Count': article['Character Count'],
            **dict(zip(reference_columns, references))
        })
    
    return pd.DataFrame(new_data)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scrape_wikipedia.py <Language_Name>")
        sys.exit(1)

    language_name = sys.argv[1]
    
    # Load the language data CSV
    lang_csv_path = "wiki_langs.csv"
    if not os.path.exists(lang_csv_path):
        print(f"Language CSV file '{lang_csv_path}' not found.")
        sys.exit(1)
    
    try:
        language_df = pd.read_csv(lang_csv_path)
    except Exception as e:
        print(f"Error reading '{lang_csv_path}': {e}")
        sys.exit(1)
    
    # Ensure required columns exist
    required_columns = {'Language', 'Language Code'}
    if not required_columns.issubset(language_df.columns):
        print(f"CSV file must contain the following columns: {required_columns}")
        sys.exit(1)
    
    # Filter to your target languages or skip if not found
    if language_name not in language_df['Language'].values:
        print(f"Language '{language_name}' not found in '{lang_csv_path}'.")
        sys.exit(1)
    
    # Get the language code
    lang_code = language_df.loc[
        language_df['Language'] == language_name, 'Language Code'
    ].values[0]
    
    print(f"Processing language: {language_name} (Code: {lang_code})")
    
    # Create directories
    save_base_path = 'wiki-articles-all'
    os.makedirs(save_base_path, exist_ok=True)
    lang_folder_name = language_name.lower()
    lang_save_path = os.path.join(save_base_path, lang_folder_name)
    os.makedirs(lang_save_path, exist_ok=True)
    
    # CSV paths
    references_csv_path = os.path.join(lang_save_path, f"wiki_references_{lang_code}.csv")
    articles_csv_path = os.path.join(lang_save_path, f"wiki_articles_{lang_code}.csv")

    # If references file already exists, skip scraping
    if os.path.exists(references_csv_path):
        print(f"References file '{references_csv_path}' already exists. Skipping scraping.")
        return
    
    # Load any existing titles to avoid duplicates
    collected_titles = load_existing_titles(articles_csv_path)
    
    # Scrape articles
    articles = scrape_wikipedia_articles(
        lang_name=language_name,
        lang_code=lang_code,
        max_articles=1000,
        max_attempts=5000,
        collected_titles=collected_titles
    )
    
    if not articles:
        print(f"No articles were scraped for language '{language_name}'.")
        sys.exit(1)
    
    # Build references
    df_with_references = build_dataset_with_references(articles)
    
    # Create a DataFrame of articles without the full content
    df_articles = pd.DataFrame([
        {
            'Article Title': a['Article Title'],
            'URL': a['URL'],
            'Word Count': a['Word Count'],
            'Character Count': a['Character Count'],
            'Category': a['Category']
        }
        for a in articles
    ])

    # Save results
    try:
        # If an articles CSV already exists, append new ones
        if os.path.exists(articles_csv_path):
            df_existing = pd.read_csv(articles_csv_path)
            df_combined = pd.concat([df_existing, df_articles], ignore_index=True)
            df_combined.to_csv(articles_csv_path, index=False, encoding='utf-8')
        else:
            df_articles.to_csv(articles_csv_path, index=False, encoding='utf-8')
        
        # Save references
        df_with_references.to_csv(references_csv_path, index=False, encoding='utf-8')
        
        print(f"Finished processing {len(df_with_references)} articles from {language_name}.")
        print(f"Data saved in '{lang_save_path}'.")
    except Exception as e:
        print(f"Error saving CSV files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
