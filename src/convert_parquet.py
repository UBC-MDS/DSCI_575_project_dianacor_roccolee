
"""
Usage:
    python pipeline.py # to convert raw downloaded jsonl.gz files to parquet, and create a sample subset for quicker iteration
"""

import argparse
from pathlib import Path
import duckdb


# Used Claude in setting up script (args) for command/bash use 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", default="../data/raw/Electronics.jsonl.gz")
    p.add_argument("--meta", default= "../data/raw/meta_Electronics.jsonl.gz")
    p.add_argument("--out-dir", default="../data/processed")
    p.add_argument("--row_limit",   type=int, default=20000, help="Max rows from each source (default: 20000)")
    p.add_argument("--subset_sample_size",  type=int, default=500,   help="Unique parent_asin keys in sample (default: 500)")
    return p.parse_args()


def main():
    args = parse_args()
    output  = Path(args.out_dir)
    reviews_raw = str(output / "reviews_raw.parquet")
    meta_raw    = str(output / "meta_raw.parquet")
    merged_out  = str(output / "merged.parquet")
    sample_out  = str(output / "sample.parquet")

    con = duckdb.connect()

# 1) convert files to parq
# 2) merge files on key
# 3) create subset 


if __name__ == "__main__":
    main()