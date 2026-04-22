import argparse
from pathlib import Path
import duckdb
from utils import read_meta_txt_columns

#reading the default meta columns from the txt file found in notebooks from EDA
default_meta_cols = read_meta_txt_columns()

def parse_args():
    """
    Serves as a centralized entry point for defining and managing the command-line 
    arguments. These arguments will be parsed and be passed directly into functions 
    being called within the script. Defaults are set for all arguments. This means 
    the script can be run without any user-specified command-line arguments.

    Returns:
        argparse.ArgumentParser:
            Configured parser instance used to define and retrieve CLI arguments.
    """
    p = argparse.ArgumentParser(description="Left-join meta and reviews Parquet files on parent_asin.")
    p.add_argument("--meta-raw",
                   default="data/raw/meta_raw.parquet",
                   help="Path to meta_raw.parquet")
    p.add_argument("--reviews-raw",
                   default="data/raw/k_reviews_raw.parquet",
                   help="Path to k_reviews_raw.parquet")
    p.add_argument("--out-dir",
                   default="data/processed",
                   help="Output directory for merged Parquet file")
    p.add_argument("--meta-cols",
                   nargs="+",
                   default=default_meta_cols,
                   help="Meta columns to include in the merge (space-separated)")
    p.add_argument("--key",
                   default="parent_asin",
                   help="Column to join on (default: parent_asin)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged_out = str(out / "merged.parquet")

    # Building columns to use in SELECT from in sql clause
    meta_cols = ", ".join(f"meta.{col}" for col in args.meta_cols)

    con = duckdb.connect()
    print("[STATUS] Connected to DuckDB. Starting merge — this may take a while...")

    # Left join so every product row is preserved even with no reviews
    con.execute(f"""
        COPY (
            SELECT meta.parent_asin, {meta_cols}, reviews.combined_reviews
            FROM read_parquet('{args.meta_raw}') AS meta
            LEFT JOIN read_parquet('{args.reviews_raw}') AS reviews USING ('{args.key}')
        )
        TO '{merged_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    print("[STATUS] Merge complete. DuckDB connection closed.")
    print(f"[SAVED] Merged files")