#!/usr/bin/env python3
"""
Process Articles Script

Converts raw JSONL articles to processed Parquet format with:
- Date filtering (>= 2025-09-01)
- Language detection
- Text length filtering (> 100 chars)
- Country mapping based on source

Usage:
    python scripts/process_articles.py
"""

import pandas as pd
import json
from pathlib import Path
from langdetect import detect
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_PATH = Path("data/raw/cfgen_articles.jsonl")
PROCESSED_DATA_PATH = Path("data/processed/cfgen_articles.parquet")

# Source to country mapping
SOURCE_TO_COUNTRY = {
    # UK sources
    'The Independent': 'uk', 'The BBC': 'uk', 'The Sun': 'uk', 'Evening Standard': 'uk',
    'Daily Express': 'uk', 'Metro': 'uk', 'Daily Star': 'uk', 'i': 'uk',
    'The Mirror': 'uk', 'Daily Mail': 'uk', 'The Guardian': 'uk',

    # US sources
    'Associated Press News': 'us', 'The Washington Times': 'us', 'Voice Of America': 'us',
    'CNBC': 'us', 'Rolling Stone': 'us', 'The Gateway Pundit': 'us', 'ABC': 'us',
    'The New Yorker': 'us', 'Wired': 'us', 'The Washington Free Beacon': 'us',
    'Vogue': 'us', 'The Nation': 'us', 'Fox News': 'us', 'The Intercept': 'us',
    'Business Insider': 'us', 'Los Angeles Times': 'us', 'TechCrunch': 'us',
}


def detect_language_batch(texts, batch_size=1000):
    """Detect language for a batch of texts using langdetect"""
    languages = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Detecting languages"):
        batch = texts[i:i+batch_size]
        for text in batch:
            try:
                lang = detect(text)
                languages.append(lang)
            except:
                languages.append('unknown')
    return languages


def main():
    logger.info(f"Loading articles from {RAW_DATA_PATH}")
    
    # Load JSONL data
    articles = pd.read_json(RAW_DATA_PATH, lines=True)
    logger.info(f"Loaded {len(articles)} articles")
    
    # Sanitize for parquet - convert dict columns to strings
    for col in articles.select_dtypes(include=['object']).columns:
        articles[col] = articles[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    
    # Convert publishing_date to datetime
    logger.info("Converting dates...")
    articles['publishing_date'] = pd.to_datetime(articles['publishing_date'], format='mixed', utc=True)
    articles['publishing_date'] = articles['publishing_date'].dt.tz_localize(None)
    
    # Filter by date (>= 2025-09-01)
    logger.info("Filtering by date (>= 2025-09-01)...")
    articles = articles[articles['publishing_date'] >= pd.Timestamp('2025-09-01')]
    logger.info(f"After date filter: {len(articles)} articles")
    
    # Filter out articles without text
    logger.info("Filtering by text presence and length...")
    articles = articles.dropna(subset=['plaintext'])
    articles = articles[articles['plaintext'].str.len() > 100]
    logger.info(f"After text filter: {len(articles)} articles")
    
    # Detect language
    logger.info("Detecting languages (this may take a while)...")
    articles['language'] = detect_language_batch(articles['plaintext'].values)
    
    # Filter to only English
    logger.info("Filtering to English articles only...")
    articles = articles[articles['language'] == 'en']
    logger.info(f"After English filter: {len(articles)} articles")
    
    # Map countries based on source
    logger.info("Mapping countries...")
    articles['country'] = articles['source'].map(SOURCE_TO_COUNTRY)
    
    # Print statistics
    logger.info("\nFinal statistics:")
    logger.info(f"Total articles: {len(articles)}")
    logger.info(f"Date range: {articles['publishing_date'].min()} to {articles['publishing_date'].max()}")
    if 'country' in articles.columns:
        logger.info("\nArticles by country:")
        logger.info(articles['country'].value_counts())
    if 'source' in articles.columns:
        logger.info("\nArticles by source:")
        logger.info(articles['source'].value_counts())
    
    # Save to parquet
    logger.info(f"\nSaving to {PROCESSED_DATA_PATH}")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(PROCESSED_DATA_PATH)
    logger.info("Done!")


if __name__ == "__main__":
    main()

