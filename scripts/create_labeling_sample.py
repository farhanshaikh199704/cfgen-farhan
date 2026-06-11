#!/usr/bin/env python3
"""
Create Labeling Sample Script

Creates two versions of a 200-claim labeling sample from generated claims:
1. With metadata (all columns including is_factual) for reference
2. For annotation (only claim and placeholder is_checkable column)

Usage:
    python scripts/create_labeling_sample.py --input <path_to_claims_csv>
    
    Or if run without arguments, it will look for the most recent claims CSV in current directory.
"""

import pandas as pd
import argparse
from pathlib import Path
import logging
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_claims_file():
    """Find the most recent claims CSV file"""
    # Look for claims files in common locations
    patterns = [
        'claims_*.csv',
        'data/samples/claims_*.csv',
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError("No claims CSV files found. Please specify --input path.")
    
    # Get the most recent file
    latest_file = max(all_files, key=lambda f: Path(f).stat().st_mtime)
    return latest_file


def main():
    parser = argparse.ArgumentParser(description='Create labeling sample from generated claims')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input claims CSV file')
    parser.add_argument('--output-dir', type=str, default='data/samples',
                        help='Directory to save output files')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Number of claims to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Find input file
    if args.input is None:
        logger.info("No input file specified, looking for most recent claims file...")
        input_file = find_latest_claims_file()
        logger.info(f"Found: {input_file}")
    else:
        input_file = args.input
    
    # Load claims
    logger.info(f"Loading claims from {input_file}")
    claims = pd.read_csv(input_file)
    logger.info(f"Loaded {len(claims)} claims")
    
    # Print distribution
    if 'is_factual' in claims.columns:
        logger.info(f"\nClaims distribution:")
        logger.info(claims['is_factual'].value_counts())
    
    # Sample claims (stratified by is_factual if possible)
    logger.info(f"\nSampling {args.sample_size} claims...")
    if 'is_factual' in claims.columns and len(claims) >= args.sample_size:
        # Stratified sampling
        sample_size_per_class = args.sample_size // 2
        factual_sample = claims[claims['is_factual'] == True].sample(
            n=min(sample_size_per_class, len(claims[claims['is_factual'] == True])),
            random_state=args.seed
        )
        non_factual_sample = claims[claims['is_factual'] == False].sample(
            n=min(sample_size_per_class, len(claims[claims['is_factual'] == False])),
            random_state=args.seed
        )
        sample = pd.concat([factual_sample, non_factual_sample]).sample(frac=1, random_state=args.seed)
        logger.info(f"Stratified sample: {len(factual_sample)} factual, {len(non_factual_sample)} non-factual")
    else:
        # Random sampling
        sample = claims.sample(n=min(args.sample_size, len(claims)), random_state=args.seed)
        logger.info(f"Random sample of {len(sample)} claims")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save version with metadata
    with_metadata_path = output_dir / "labeling_sample_200_with_metadata.csv"
    logger.info(f"\nSaving sample with metadata to {with_metadata_path}")
    sample.to_csv(with_metadata_path, index=False)
    
    # Create version for annotation (only claim and is_checkable placeholder)
    for_annotation = pd.DataFrame({
        'claim': sample['claim'],
        'is_checkable': '',  # Empty for annotators to fill
        'notes': ''  # Optional notes column
    })
    
    for_annotation_path = output_dir / "labeling_sample_200_for_annotation.csv"
    logger.info(f"Saving sample for annotation to {for_annotation_path}")
    for_annotation.to_csv(for_annotation_path, index=False)
    
    logger.info("\nDone!")
    logger.info(f"Files created:")
    logger.info(f"  1. {with_metadata_path} (all columns including is_factual)")
    logger.info(f"  2. {for_annotation_path} (claim + empty is_checkable)")


if __name__ == "__main__":
    main()

