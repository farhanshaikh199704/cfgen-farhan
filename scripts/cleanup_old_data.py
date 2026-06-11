#!/usr/bin/env python3
"""
Cleanup Script

Deletes old data files and claim generations that are no longer needed.
Run this AFTER completing the new claim generation and verification.

Usage:
    python scripts/cleanup_old_data.py [--dry-run]
"""

from pathlib import Path
import logging
import glob
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Clean up old data files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    args = parser.parse_args()

    logger.info("Starting cleanup..." + (" (dry run)" if args.dry_run else ""))

    # Patterns to delete (glob patterns)
    patterns_to_delete = [
        # Intermediate claim files in root directory (legacy location)
        "claims_*_intermediate_*.csv",
        # Intermediate claim files in new location
        "data/intermediates/claims_*_intermediate_*.csv",
        # Intermediate claim files in data/samples (old location)
        "data/samples/claims_*_intermediate_*.csv",
        # Backup parquet files
        "data/processed/*_backup*.parquet",
        # Temp parquet files
        "data/processed/*_temp*.parquet",
    ]

    total_deleted = 0
    total_size = 0

    for pattern in patterns_to_delete:
        files = glob.glob(pattern)
        for file_path in files:
            path = Path(file_path)
            try:
                size = path.stat().st_size
                if args.dry_run:
                    logger.info(f"Would delete: {path} ({size / 1024 / 1024:.1f} MB)")
                else:
                    path.unlink()
                    logger.info(f"Deleted: {path} ({size / 1024 / 1024:.1f} MB)")
                total_deleted += 1
                total_size += size
            except Exception as e:
                logger.error(f"Failed to delete {path}: {e}")

    action = "Would delete" if args.dry_run else "Deleted"
    logger.info(f"Cleanup completed! {action} {total_deleted} files ({total_size / 1024 / 1024 / 1024:.2f} GB)")


if __name__ == "__main__":
    main()
