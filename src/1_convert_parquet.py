"""
Usage:
    python 1_convert_parquet.py \
        --reviews data/raw/Electronics.jsonl.gz \
        --meta data/raw/meta_Electronics.jsonl.gz \
        --out-dir data/raw

    References:
    ilyamusabirov/525eda_duckdb.md: https://gist.github.com/ilyamusabirov/9491e5ce6ae2fc63d6222609cebd0588
"""

import argparse
from pathlib import Path
import duckdb


def parse_args():
    '''To accept arguments directly via bash/terminal commands'''
    p = argparse.ArgumentParser(description="Convert raw JSONL.GZ files to Parquet.")
    p.add_argument("--reviews-data",
                   default="data/raw/Electronics.jsonl.gz",
                   help="Path to raw reviews JSONL.GZ file")
    p.add_argument("--meta-data",
                   default="data/raw/meta_Electronics.jsonl.gz",
                   help="Path to raw meta JSONL.GZ file")
    p.add_argument("--out-dir",
                   default="data/raw",
                   help="Output directory for Parquet files")
    p.add_argument("--top-k-reviews", type=int,
                   default=10,
                   help="Max number of top reviews (by helpful_vote) to aggregate per product")
    return p.parse_args()


def main():
    ''' Main script function loop to convert files to parquet. '''
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_parquet_path = str(out / "meta_raw.parquet")
    reviews_parquet_path = str(out / "k_reviews_raw.parquet")

    con = duckdb.connect()
    print("[STATUS] Connected to DuckDB.")

    # --- Meta Source & conversion ---
    con.execute(f"""
        COPY (
            SELECT * 
            FROM read_json_auto('{args.meta_data}', ignore_errors=true)
        )
        TO '{meta_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print("[DONE] Meta data exported as full-sized Parquet.")

    # --- Reviews 
    # aggregates/concatenates all reviews into 1 row per product, 
    # Takes only top N reviews via argument ---
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT
                    parent_asin,
                    title,
                    helpful_vote,
                    ROW_NUMBER() OVER (
                        PARTITION BY parent_asin
                        ORDER BY helpful_vote DESC
                    ) AS rn
                FROM read_json_auto('{args.reviews_data}', ignore_errors=true)
            )
            SELECT
                parent_asin,
                STRING_AGG(title, ' | ' ORDER BY helpful_vote DESC) AS combined_reviews
            FROM ranked
            WHERE rn <= {args.top_k_reviews}
            GROUP BY parent_asin
        )
        TO '{reviews_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[DONE] Reviews exported (with only the top {args.top_reviews} reviews as 1 row per product).")

    con.close()
    print("[STATUS] Conversion complete. DuckDB connection closed.")
    print(f"[SAVED] Raw meta as parquet")
    print(f"[SAVED] Raw reviews as parquet")


if __name__ == "__main__":
    main()