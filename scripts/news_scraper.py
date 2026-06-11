#!/usr/bin/env python3
"""
News Scraper with External Multithreading

Uses ThreadPoolExecutor to scrape individual publishers in parallel,
with each Crawler running in single-threaded mode to avoid fundus internal threading bugs.
"""

import json
from pathlib import Path
from datetime import datetime, date
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from fundus import PublisherCollection, Crawler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output file path
OUTPUT_FILE = Path("data/raw/cfgen_articles.jsonl")

# Thread lock for file writing
file_lock = threading.Lock()

# Global article counter
article_counter = {'count': 0}
counter_lock = threading.Lock()

# Define a date filter for articles published after September 1, 2025
def date_filter(article) -> bool:
    """Filter to include only articles published on or after September 1, 2025."""
    start_date = date(2025, 9, 1)
    
    if hasattr(article, 'publishing_date') and article.publishing_date:
        try:
            pub_date = article.publishing_date
            if isinstance(pub_date, datetime):
                pub_date = pub_date.date()
            return pub_date >= start_date
        except (ValueError, TypeError, AttributeError):
            return False
    return False


def save_article(article):
    """Thread-safe article saving"""
    try:
        article_json = {
            "publishing_date": article.publishing_date.isoformat() if article.publishing_date else None,
            "plaintext": article.plaintext, 
            "title": article.title,
            "url": article.html.requested_url,
            "authors": article.authors,
            "source": article.publisher,
            "topics": article.topics,
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with file_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(article_json) + "\n")
        
        with counter_lock:
            article_counter['count'] += 1
            if article_counter['count'] % 100 == 0:
                logger.info(f"Total saved: {article_counter['count']} articles")
        
        return True
    except Exception as e:
        logger.error(f"Error saving article: {str(e)}")
        return False


def scrape_single_publisher(publisher):
    """Scrape a single publisher with threading=False to avoid internal fundus bugs"""
    pub_name = getattr(publisher, 'name', str(publisher))
    saved_count = 0
    
    try:
        # Use threading=False for each individual crawler to avoid fundus threading bugs
        crawler = Crawler(publisher, threading=False)
        
        for article in crawler.crawl(max_articles=10000, only_complete=True, error_handling='catch'):
            if date_filter(article):
                if save_article(article):
                    saved_count += 1
                    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning(f"Error scraping {pub_name}: {e}")
    
    logger.info(f"Finished {pub_name}: saved {saved_count} articles")
    return saved_count


def get_all_publishers():
    """Get list of all individual US and UK publishers"""
    publishers = []
    
    # Get individual US publishers
    for pub in PublisherCollection.us:
        publishers.append(pub)
    
    # Get individual UK publishers  
    for pub in PublisherCollection.uk:
        publishers.append(pub)
    
    return publishers


def main():
    logger.info("Starting scraper with date filter >= 2025-09-01")
    logger.info("Using external ThreadPoolExecutor with single-threaded crawlers per publisher")
    
    # Get all individual publishers
    publishers = get_all_publishers()
    logger.info(f"Found {len(publishers)} individual publishers to scrape")
    
    # Use ThreadPoolExecutor to scrape publishers in parallel
    # Each crawler runs single-threaded internally to avoid fundus bugs
    max_workers = 8  # Number of parallel publisher scrapers
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all publishers for scraping
            future_to_pub = {executor.submit(scrape_single_publisher, pub): pub for pub in publishers}
            
            # Process results as they complete
            for future in as_completed(future_to_pub):
                pub = future_to_pub[future]
                pub_name = getattr(pub, 'name', str(pub))
                try:
                    count = future.result()
                except Exception as e:
                    logger.error(f"Publisher {pub_name} generated an exception: {e}")
        
        logger.info(f"Scraping completed. Total articles saved: {article_counter['count']}")
        
    except KeyboardInterrupt:
        logger.info(f"Scraper interrupted. Total saved: {article_counter['count']} articles")
    except Exception as e:
        logger.error(f"Main error: {e}")
        logger.info(f"Total saved before error: {article_counter['count']} articles")


if __name__ == "__main__":
    main()
