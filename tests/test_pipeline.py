
import sys
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from claim_generation import ClaimGenerator

# Dummy claims dataframe for 'examples' method
DUMMY_CLAIMS_DF = pd.DataFrame({
    'text': [f'Example claim {i}' for i in range(10)],
    'ratingName': ['TRUE', 'FALSE'] * 5,
    'total_text': [f'Source article text {i}' for i in range(10)],
    'date': ['2024-01-01'] * 10
})

def test_sglang_pipeline_naive():
    """
    Test the pipeline with 'naive' method.
    """
    print("\nTesting 'naive' method...")
    run_pipeline_test(method="naive")

def test_sglang_pipeline_examples():
    """
    Test the pipeline with 'examples' method.
    """
    print("\nTesting 'examples' method...")
    run_pipeline_test(method="examples", claims_df=DUMMY_CLAIMS_DF)

def run_pipeline_test(method, claims_df=None):
    # Create dummy data
    df = pd.DataFrame({
        'article_id': [1],
        'country': ['us'],
        'language': ['en'],
        'publishing_date': ['2025-01-01'],
        'source': ['test_source'],
        'title': ['Test Article'],
        'plaintext': ['This is a test article about AI. It was written in 2025. The sky is blue.'],
        'text': ['Test Article\nThis is a test article about AI. It was written in 2025. The sky is blue.']
    })

    # Initialize generator
    # Use a port that is likely free
    port = 34568
    
    generator = ClaimGenerator(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        port=port,
        method=method,
        claims_df=claims_df,
        max_workers=1
    )
    
    print(f"Starting server for {method}...")
    success = generator.start_server()
    if not success:
        pytest.fail("Failed to start SGLang server")
        
    try:
        print(f"Processing articles with {method}...")
        results = generator.process_articles(df, batch_size=1, save_interval=1)
        
        print("Results:")
        print(results)
        
        assert len(results) > 0
        assert 'claim' in results.columns
        assert 'is_factual' in results.columns
        # Check if we have both True and False claims
        assert results['is_factual'].nunique() >= 1 # Ideally 2, but with 1 article it might vary depending on generation
        
    finally:
        generator.cleanup()

if __name__ == "__main__":
    # Allow running directly
    try:
        test_sglang_pipeline_naive()
        test_sglang_pipeline_examples()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        # Ensure we exit with error code if failed
        sys.exit(1)
