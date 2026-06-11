# claim_generation.py
"""
Claim Generation Script

This script generates factual and non-factual claims from news articles using LLM.
It supports two methods of generation:
1. 'naive' - Direct generation without examples
2. 'examples' - Generation with examples from a claims dataset

Requirements:
- Python 3.8+
- Required packages: pandas, numpy, dspy, tqdm, sglang[all]
- SGLang server running locally
- Input files:
  * cfgen_articles.parquet: News articles dataset
  * merged_GESIS_total.csv: Claims dataset (only needed for 'examples' method)

Usage:
1. First ensure SGLang server is running (default port 7501)

2. Run the script with required arguments:
   ```
   # Basic usage with naive method:
   python claim_generation.py --method naive

   # Using examples method:
   python claim_generation.py --method examples

   # Specify custom port for SGLang server:
   python claim_generation.py --method naive --port 7502

   # Specify custom output file:
   python claim_generation.py --method naive --output my_claims.csv
   ```

Output:
- CSV file with columns:
  * article_id: ID of source article
  * country: Country of source article
  * language: Language of article
  * date: Publication date
  * article_text: Full text of source article
  * claim: Generated claim
  * is_factual: True/False indicating if claim is factual
  * reasoning: Model's reasoning for generating claims

- Intermediate results are saved every 10 articles
- Error states are saved if exceptions occur
"""

import pandas as pd
import numpy as np
import dspy
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import time
import random
import logging
import os
import concurrent.futures
from typing import Dict, Any, Optional
import re
from pathlib import Path
import weave
import subprocess
import atexit
import psutil
import sys

# Import SGLang utilities
try:
    from sglang.test.test_utils import is_in_ci
    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd
    from sglang.utils import wait_for_server, terminate_process
except ImportError:
    print("Warning: SGLang utilities not found. Using fallback implementation.")
    launch_server_cmd = None
    wait_for_server = None
    terminate_process = None

# Configure logging - reduce level to WARNING to show fewer messages
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("claim_generation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Disable weave call link printing
os.environ["WEAVE_PRINT_CALL_LINK"] = "False"

# weave.init(project_name="cfgen")



# System prompt
system_prompt = """
You are an expert fact generator. You are given a news article and, based on the content, you will generate a few verifiable factual claims.
"""

# Simple prompt template
simple_prompt = """
Based on the following article, generate between 1 and 3 verifiable factual claims and between 1 and 3 non-factual claims.
The factual claims should be clearly true based on the content of the article, while the non-factual claims should be false and in contrast to the content of the article.
Each claim should be able to be verified on its own and should therefore contain all the context around the claim. No claim should be vague or ambiguous or need information from the article or the other claims to be understood. Repeat the context needed in each claim, like the time of the event, the location, the person, etc. The claims should be in the same language as the article. 

Requirements for each claim:
- the claim should be verifiable on its own, without having access to the article or the other claims
- the claim should be in the same language as the article
- the claim should be clear and not ambiguous
- the claim should contain all the context needed to be understood
- the claim should be plausible
- the timeframe (and if necessary the location) should be understandable from the claim alone (if not specified in the article, use the publishing date, like "As of 2024-01-01, ...")

Before generating the claims, reason about:
- the plausibility of the claims you can generate
- the number of true and false claims you can generate
- the mix of true and false claims you can generate
- the details you should include in each claim to make it verifiable
- the draft of the claims you will generate
- checking the plausibility of the claims you will generate
- checking if the claims contain all the information needed to be verifiable, most importantly a specific timeframe 

Your response MUST follow this EXACT format:

<REASONING>
Your detailed reasoning here...
</REASONING>

<FACTUAL_CLAIMS>
- factual claim 1
- factual claim 2
- factual claim 3
</FACTUAL_CLAIMS>

<NON_FACTUAL_CLAIMS>
- non-factual claim 1
- non-factual claim 2
- non-factual claim 3
</NON_FACTUAL_CLAIMS>

Generate between 1-3 claims in EACH category. You MUST generate at least one claim in each category.

Date of publication:
{date}
Article:
{article}
"""

def parse_claims(answer):
    """Parse the LLM response into reasoning, factual and non-factual claims using regex patterns"""
    # Define regex patterns for each section with the new delimiters
    reasoning_pattern = r'<REASONING>(.*?)</REASONING>'
    factual_pattern = r'<FACTUAL_CLAIMS>(.*?)</FACTUAL_CLAIMS>'
    non_factual_pattern = r'<NON_FACTUAL_CLAIMS>(.*?)</NON_FACTUAL_CLAIMS>'
    
    # Default empty values
    reasoning = ""
    factual_claims = []
    non_factual_claims = []
    
    # Extract reasoning
    reasoning_match = re.search(reasoning_pattern, answer, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract factual claims
    factual_match = re.search(factual_pattern, answer, re.DOTALL)
    if factual_match:
        factual_text = factual_match.group(1).strip()
        # Extract individual claims (lines starting with - or numbered)
        for line in factual_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or re.match(r'^\d+\.', line)):
                claim = re.sub(r'^-|\d+\.\s*', '', line).strip()
                if claim:
                    factual_claims.append(claim)
    
    # Extract non-factual claims
    non_factual_match = re.search(non_factual_pattern, answer, re.DOTALL)
    if non_factual_match:
        non_factual_text = non_factual_match.group(1).strip()
        # Extract individual claims (lines starting with - or numbered)
        for line in non_factual_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or re.match(r'^\d+\.', line)):
                claim = re.sub(r'^-|\d+\.\s*', '', line).strip()
                if claim:
                    non_factual_claims.append(claim)
    
    # Fallback parsing if the regex approach failed to find any claims
    if not factual_claims or not non_factual_claims:
        # Try the original parsing approach as fallback
        return fallback_parse_claims(answer)
    
    return reasoning, factual_claims, non_factual_claims

def fallback_parse_claims(answer):
    """Fallback parser for when structured format parsing fails"""
    reasoning = ""
    factual_claims = []
    non_factual_claims = []
    current_section = None
    
    # Various section headers we might encounter
    factual_headers = ['factual claims', 'true claims', 'factual', 'true']
    non_factual_headers = ['non-factual claims', 'false claims', 'non-factual', 'false']
    reasoning_headers = ['reasoning', 'rationale', 'analysis']
    
    for line in answer.split("\n"):
        line = line.strip()
        
        # Check for section headers
        line_lower = line.lower()
        
        if any(header in line_lower for header in reasoning_headers):
            current_section = "reasoning"
            continue
        elif any(header in line_lower for header in factual_headers):
            current_section = "factual"
            continue
        elif any(header in line_lower for header in non_factual_headers):
            current_section = "non-factual"
            continue
            
        # Process content based on current section
        if not line or not current_section:
            continue
            
        if current_section == "reasoning":
            reasoning += line + "\n"
        elif line.startswith(("- ", "• ", "* ", "1. ", "2. ", "3. ")):
            # Handle various bullet point and numbered list formats
            claim = re.sub(r'^[•\-\*]\s+|^\d+\.\s+', '', line).strip()
            if current_section == "factual":
                factual_claims.append(claim)
            elif current_section == "non-factual":
                non_factual_claims.append(claim)
    
    return reasoning.strip(), factual_claims, non_factual_claims

@weave.op(name="generate_claims_sync")
def generate_claims_sync(article, date, lm):
    """Synchronous version of generate_claims for threading"""
    prompt = simple_prompt.format(article=article, date=date)
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = lm(messages=conversation, cache=False, temperature=1)
    answer = response.content if hasattr(response, 'content') else response
    reasoning, factual_claims, non_factual_claims = parse_claims(answer[0])
    return date, reasoning, factual_claims, non_factual_claims

@weave.op(name="generate_claims_with_examples_sync")
def generate_claims_with_examples_sync(article, date, claims_df, lm):
    """Synchronous version of generate_claims_with_examples for threading"""
    # Select 3 random articles
    articles = pd.Series(claims_df.total_text.unique()).sample(3)
    
    # Add example claims to the prompt
    examples_text = "Here are some example claims:\n"
    for i, article_text in enumerate(articles, 1):
        examples_text += f"{i}. Article: {article_text}\n"
        # Get all claims and ratings for this article
        claims = claims_df[claims_df.total_text == article_text][['text', 'ratingName']].drop_duplicates()
        for _, claim_row in claims.iterrows():
            examples_text += f"Claim: {claim_row['text']}\nRating: {claim_row['ratingName']}\n"
        examples_text += "\n"
    
    # Add the article to the prompt
    prompt = examples_text + simple_prompt.format(article=article, date=date)
    
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = lm(messages=conversation, cache=False, temperature=1)
    answer = response.content if hasattr(response, 'content') else response
    reasoning, factual_claims, non_factual_claims = parse_claims(answer[0])
    return date, reasoning, factual_claims, non_factual_claims

# Define server startup function at module level so it can be pickled
def _start_server_process(base_cmd):
    from sglang.utils import launch_server_cmd, wait_for_server
    process, port = launch_server_cmd(base_cmd)
    wait_for_server(f"http://localhost:{port}")
    return process, port

class ClaimGenerator:
    def __init__(self, model_name: str, port: int, method: str, claims_df=None, max_workers: int = 50, 
                 retry_attempts: int = 3, retry_delay: float = 2.0, tensor_parallel: int = 1, data_parallel: int = 1):
        """Initialize the claim generator with required components and settings.
        
        Args:
            model_name: Name of the model to use (e.g. "meta-llama/Llama-3.1-70B-Instruct")
            port: Port for the SGLang server
            method: The method of claim generation ('naive' or 'examples')
            claims_df: DataFrame containing examples (only needed for 'examples' method)
            max_workers: Maximum number of concurrent threads
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
            tensor_parallel: Number of GPUs to use for tensor parallelism
        """
        self.model_name = model_name
        self.port = port
        self.method = method
        self.claims_df = claims_df
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.tensor_parallel = tensor_parallel
        self.data_parallel = data_parallel
        self.results = []
        self.errors = []
        self.article_count = 0
        self.successful_count = 0
        self.server_process = None
        self.lm = None
        
    def start_server(self):
        """Start the SGLang server with the specified model and configuration"""
        try:
            # Kill any existing process on the port
            self._kill_process_on_port(self.port)
            
            # Set CUDA devices based on tensor parallelism (only if not already set externally)
            if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
                if self.tensor_parallel > 1 or self.data_parallel > 1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                logger.info(f"Using pre-set CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # Construct the base command
            base_cmd = f"{sys.executable} -m sglang.launch_server --model {self.model_name}"
            if self.tensor_parallel > 1:
                base_cmd += f" --tp {self.tensor_parallel}"
            if self.data_parallel > 1:
                base_cmd += f" --dp {self.data_parallel}"
            base_cmd += f" --port {self.port}"
            
            # Use SGLang's utilities if available, otherwise fallback
            if launch_server_cmd and wait_for_server:
                logger.info("Using SGLang utilities for server management")
                
                # Run server startup in separate process
                # Start server process directly without ProcessPoolExecutor
                try:
                    self.server_process, actual_port = _start_server_process(base_cmd)
                except Exception as e:
                    logger.error(f"Server startup failed: {str(e)}")
                    return False
                # Update port to match actual port
                self.port = actual_port
                logger.info(f"Server assigned to port {self.port}")
                
                # Initialize the LM client with the correct port
                try:
                    self.lm = dspy.LM(
                        f"openai/{self.model_name}",
                        api_base=f"http://localhost:{self.port}/v1",
                        api_key="local",
                        model_type='chat',
                        max_tokens=8096
                    )
                    dspy.configure(lm=self.lm)
                    logger.info(f"LM client initialized with port {self.port}")
                    print(f"LM client initialized with port {self.port}, starting to generate claims")
                    
                    # Register cleanup handler
                    atexit.register(self.cleanup)
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to initialize LM client: {str(e)}")
                    self.cleanup()
                    return False
            else:
                # Fallback to manual process management
                logger.warning("Using fallback server management")
                cmd = base_cmd.split()
                cmd.extend(["--port", str(self.port)])
                
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Wait for server to start
                time.sleep(30)  # Increased wait time for larger models
                
                # Initialize the LM client
                try:
                    self.lm = dspy.LM(
                        self.model_name,
                        api_base=f"http://localhost:{self.port}/v1",
                        api_key="local",
                        model_type='chat',
                        max_tokens=8096
                    )
                    dspy.configure(lm=self.lm)
                    logger.info(f"LM client initialized with port {self.port}")
                    
                    # Register cleanup handler
                    atexit.register(self.cleanup)
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to initialize LM client: {str(e)}")
                    self.cleanup()
                    return False
            
        except Exception as e:
            logger.error(f"Failed to start SGLang server: {str(e)}")
            self.cleanup()
            return False
    
    def _kill_process_on_port(self, port):
        """Kill any existing process running on the specified port"""
        try:
            # Find process using the port
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr.port == port:
                            logger.warning(f"Killing existing process on port {port}")
                            proc.terminate()
                            proc.wait(timeout=5)
                            return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.warning(f"Error while trying to kill process on port {port}: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.server_process:
                logger.info("Stopping SGLang server...")
                if terminate_process:
                    # Use SGLang's terminate_process if available
                    terminate_process(self.server_process)
                else:
                    # Fallback cleanup
                    self.server_process.terminate()
                    try:
                        self.server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.server_process.kill()
                self.server_process = None
            
            # Cleanup port
            self._kill_process_on_port(self.port)
            
            # Clear LM
            self.lm = None
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def _process_single_article(self, i: int, article: str, date: str) -> Optional[Dict[str, Any]]:
        """Process a single article with retries"""
        for attempt in range(self.retry_attempts):
            try:
                if self.method == 'naive':
                    date_str, reasoning, factual_claims, non_factual_claims = generate_claims_sync(
                        article, date, self.lm)
                else:  # examples
                    date_str, reasoning, factual_claims, non_factual_claims = generate_claims_with_examples_sync(
                        article, date, self.claims_df, self.lm)
                
                # Stronger validation
                if not factual_claims:
                    logger.warning(f"No factual claims generated for article {i}, retrying...")
                    raise ValueError("No factual claims generated")
                    
                if not non_factual_claims:
                    logger.warning(f"No non-factual claims generated for article {i}, retrying...")
                    raise ValueError("No non-factual claims generated")
                    
                # Filter out empty or very short claims
                factual_claims = [c for c in factual_claims if len(c) > 10]
                non_factual_claims = [c for c in non_factual_claims if len(c) > 10]
                
                # Still check again after filtering
                if not factual_claims or not non_factual_claims:
                    logger.warning(f"Invalid claims for article {i} after filtering, retrying...")
                    raise ValueError("Invalid claims after filtering")
                
                self.successful_count += 1

                return {
                    'article_id': i,
                    'country': self.articles_sample.country.iloc[i],
                    'language': self.articles_sample.language.iloc[i],
                    'date': str(date),
                    'article_text': article,
                    'source': self.articles_sample.source.iloc[i],  # Add source information
                    'reasoning': reasoning,
                    'factual_claims': factual_claims,
                    'non_factual_claims': non_factual_claims
                }
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    # Add jitter to retry delay to avoid thundering herd
                    delay = self.retry_delay * (1 + 0.2 * random.random())
                    logger.warning(f"Error on article {i} (attempt {attempt+1}/{self.retry_attempts}): {str(e)[:50]}...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed article {i}: {str(e)[:100]}")
                    self.errors.append({'article_id': i, 'error': str(e)})
                    return None
    
    def process_articles(self, articles_sample: pd.DataFrame, 
                        batch_size: int = 50, 
                        save_interval: int = 10) -> pd.DataFrame:
        """Process all articles in parallel using a thread pool"""
        self.articles_sample = articles_sample
        self.article_count = len(articles_sample)
        start_time = time.time()
        
        # Create a progress bar
        pbar = tqdm(total=self.article_count, desc="Processing articles")
        
        # Process in batches to manage memory and provide status updates
        for batch_start in range(0, self.article_count, batch_size):
            batch_end = min(batch_start + batch_size, self.article_count)
            batch_indices = range(batch_start, batch_end)
            
            # Process this batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(
                        self._process_single_article, 
                        i, 
                        articles_sample.text.iloc[i], 
                        articles_sample.publishing_date.iloc[i]
                    ): i for i in batch_indices
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Unhandled error in thread for article {i}: {str(e)}")
                    finally:
                        # Update progress regardless of success/failure
                        pbar.update(1)
                        
                        # Calculate and display statistics
                        elapsed = time.time() - start_time
                        avg_time = elapsed / max(1, pbar.n)
                        success_rate = 100 * self.successful_count / max(1, pbar.n)
                        remaining = avg_time * (self.article_count - pbar.n)
                        
                        pbar.set_postfix({
                            'success': f"{success_rate:.1f}%", 
                            'avg': f"{avg_time:.2f}s", 
                            'eta': f"{remaining/60:.1f}m"
                        })
            
            # Save intermediate results at intervals
            if batch_end % save_interval == 0 or batch_end == self.article_count:
                self._save_intermediate_results()
                logger.info(f"Saved intermediate results at batch {batch_end}/{self.article_count}")
        
        pbar.close()
        
        # Save error log if any
        if self.errors:
            error_file = f"claim_generation_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w') as f:
                json.dump(self.errors, f, indent=2)
            logger.info(f"Saved error log to {error_file}")
        
        # Return final results DataFrame
        return format_claims_for_csv(self.results)
    
    def _save_intermediate_results(self):
        """Save intermediate results to a dedicated directory."""
        if not self.results:
            logger.warning("No results to save as intermediate results")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_df = format_claims_for_csv(self.results)

        # Save to dedicated intermediates directory
        intermediate_dir = Path("data/intermediates")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        intermediate_output = intermediate_dir / f"claims_{self.method}_intermediate_{timestamp}.csv"

        try:
            intermediate_df.to_csv(intermediate_output, index=False)
            logger.info(f"Saved {len(intermediate_df)} intermediate claims to {intermediate_output}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")

def format_claims_for_csv(results):
    """Convert results into a format suitable for CSV output"""
    rows = []
    for result in results:
        article_id = result['article_id']
        country = result['country']
        language = result['language']
        date = result['date']
        article_text = result['article_text']
        source = result.get('source', 'unknown')  # Get source with fallback
        
        # Add factual claims
        for claim in result['factual_claims']:
            rows.append({   
                'article_id': article_id,
                'country': country,
                'language': language,
                'date': date,
                'article_text': article_text,
                'source': source,  # Include source in each row
                'claim': claim,
                'is_factual': True,
                'reasoning': result['reasoning']
            })
            
        # Add non-factual claims
        for claim in result['non_factual_claims']:
            rows.append({
                'article_id': article_id,
                'country': country,
                'language': language,
                'date': date,
                'article_text': article_text,
                'source': source,  # Include source in each row
                'claim': claim,
                'is_factual': False,
                'reasoning': result['reasoning']
            })
            
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Generate claims from articles')
    parser.add_argument('--method', type=str, choices=['naive', 'examples'], required=True,
                        help='Method to use for generation: naive or examples')
    parser.add_argument('--port', type=int, default=7501, help='Port for the SGLang server')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--max_workers', type=int, default=100, 
                        help='Maximum number of concurrent worker threads')
    parser.add_argument('--batch_size', type=int, default=50, 
                        help='Number of articles to process in each batch')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Number of articles after which to save intermediate results')
    parser.add_argument('--retry_attempts', type=int, default=10,
                        help='Number of retry attempts for failed requests')
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B",
                        help='Model to use for generation')
    parser.add_argument('--tensor_parallel', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--use_all_articles', action='store_true',
                        help='Use all articles from the selected countries instead of sampling')
    parser.add_argument('--data_parallel', type=int, default=1,
                        help='Number of GPUs to use for data parallelism')
    parser.add_argument('--num_articles', type=int, default=25,
                        help='Number of articles to sample (default: 25, ignored if --use_all_articles)')
    parser.add_argument('--input', type=str, default=None,
                        help='Custom input parquet file (skips all filtering/sampling if provided)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='CUDA_VISIBLE_DEVICES to use (e.g., "1,3"). Overrides auto-detection.')
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES early if specified (before any CUDA operations)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using specified GPUs: {args.gpus}")
    
    # Load data
    print("Loading articles...")

    if args.input:
        # Use custom pre-filtered input file (skips all filtering/sampling)
        print(f"Loading custom input file: {args.input}")
        articles_sample = pd.read_parquet(args.input)
        articles_sample = articles_sample[articles_sample["plaintext"].notna()]
        articles_sample["text"] = articles_sample["title"] + "\n" + articles_sample["plaintext"]
        # Reset index to ensure article_id in output is 0-indexed
        articles_sample = articles_sample.reset_index(drop=True)
        print(f"Loaded {len(articles_sample)} articles from custom input")
    else:
        articles = pd.read_parquet('data/processed/cfgen_articles.parquet')
        articles = articles[articles["plaintext"].notna()]
        articles["text"] = articles["title"] + "\n" + articles["plaintext"]

        # Filter to English articles only
        np.random.seed(42)
        articles = articles[articles['language'] == 'en']
        articles_sample = articles[articles['country'] == 'us']  # US articles only

        # Filter to September and October 2025 only (before November)
        print("Filtering to US articles from September-October 2025 only...")
        articles_sample = articles_sample[
            (articles_sample['publishing_date'] >= pd.Timestamp('2025-09-01')) &
            (articles_sample['publishing_date'] < pd.Timestamp('2025-11-01'))
        ]
        print(f"After date filter (US, Sep-Oct 2025): {len(articles_sample)} articles")

        # Sample articles or use all articles
        if not args.use_all_articles:
            print(f"Sampling {args.num_articles} articles...")
            articles_sample = articles_sample.sample(n=min(args.num_articles, len(articles_sample)), random_state=42)
        else:
            print(f"Using all articles from selected countries. Total: {len(articles_sample)}")
    
    # Load claims if using examples method
    claims_df = None
    if args.method == 'examples':
        print("Loading claims dataset...")
        claims_df = pd.read_csv('data/raw/merged_GESIS_total.csv')
        claims_df = claims_df[claims_df.ratingName.isin(['TRUE', 'FALSE'])]
    
    # Initialize claim generator
    claim_generator = ClaimGenerator(
        model_name=args.model,
        port=args.port,
        method=args.method,
        claims_df=claims_df,
        max_workers=args.max_workers,
        retry_attempts=args.retry_attempts,
        tensor_parallel=args.tensor_parallel,
        data_parallel=args.data_parallel
    )
    
    # Start the server
    print(f"Starting SGLang server for {args.model}...")
    if not claim_generator.start_server():
        print("Failed to start SGLang server. Exiting.")
        return
    
    try:
        # Process all articles
        print(f"Starting claim generation using {args.method} method with {args.max_workers} workers...")
        start_time = time.time()
        
        results_df = claim_generator.process_articles(
            articles_sample=articles_sample,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
        
        # Calculate and log statistics
        total_time = time.time() - start_time
        print(f"Completed: {claim_generator.successful_count}/{claim_generator.article_count} articles in {total_time/60:.1f}m")
        print(f"Success rate: {100 * claim_generator.successful_count / claim_generator.article_count:.1f}%")
        
        # Save final results
        output_file = args.output if args.output else f"claims_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Clean up intermediate files after successful completion
        intermediate_dir = Path("data/intermediates")
        if intermediate_dir.exists():
            deleted_count = 0
            for f in intermediate_dir.glob(f"claims_{args.method}_intermediate_*.csv"):
                f.unlink()
                deleted_count += 1
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} intermediate files")

    finally:
        # Ensure cleanup happens
        claim_generator.cleanup()

if __name__ == "__main__":
    main()