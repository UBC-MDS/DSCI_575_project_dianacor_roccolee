"""
Usage:
    python 3_export_subsets.py
    python 3_export_subsets.py \
        --meta-raw data/raw/meta_raw.parquet \
        --reviews-raw data/raw/k_reviews_raw.parquet \
        --merged data/processed/merged.parquet \
        --out-dir data/processed \
        --sample-size 10000
"""
import argparse
from pathlib import Path
import duckdb

def parse_args():
    '''To accept arguments directly via bash/terminal commands'''
    p = argparse.ArgumentParser(description="Export row-limited subsets of Parquet files for faster iteration.")
    p.add_argument("--meta-raw",
                   default="data/raw/meta_raw.parquet",
                   help="Path to meta_raw.parquet")
    p.add_argument("--reviews-raw",
                   default="data/raw/k_reviews_raw.parquet",
                   help="Path to k_reviews_raw.parquet")
    p.add_argument("--merged",
                   default="data/processed/merged.parquet",
                   help="Path to merged.parquet")
    p.add_argument("--raw-out-dir",
                   default="data/raw",
                   help="Output directory for subset of the raw Parquet files")
    p.add_argument("--merged-out-dir",
                   default="data/processed",
                   help="Output directory for subset of the merged Parquet file")
    p.add_argument("--sample-size", type=int,
                   default=10000,
                   help="Row limit for subsets")
    return p.parse_args()

def main():
    ''' Main script function loop to export only a subset of files. 
    Will be added to repo for those who have limited compute resources 
    and/or want to iterate through the rest of the workflow. '''

    args = parse_args()
    raw_out = Path(args.raw_out_dir)
    merge_out = Path(args.merged_out_dir)
    raw_out.mkdir(parents=True, exist_ok=True)
    merge_out.mkdir(parents=True, exist_ok=True)

    meta_subset    = str(raw_out / "meta_raw_subset.parquet")
    reviews_subset = str(raw_out / "k_reviews_raw_subset.parquet")
    merged_subset  = str(merge_out / "merged_subset.parquet")

    con = duckdb.connect()
    print("[STATUS] Connected to DuckDB. Exporting subsets...")

    # --- Meta subset (first K rows) ---
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{args.meta_raw}')
            LIMIT {args.sample_size}
        )
        TO '{meta_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[SAVED] Meta subset exported.")

    # --- Reviews subset (first K rows) ---
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{args.reviews_raw}')
            LIMIT {args.sample_size}
        )
        TO '{reviews_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[SAVED] Reviews subset exported.")

    # --- Merged subset ---
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{args.merged}')
            LIMIT {args.sample_size}
        )
        TO '{merged_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[SAVED] Merged subset exported")

    con.close()
    print("[STATUS] All subsets exported. DuckDB connection closed.")

if __name__ == "__main__":
    main()