
"""
Usage:
    python pipeline.py # to convert raw downloaded jsonl.gz files to parquet, and create a sample subset for quicker iteration
References: 
    ilyamusabirov/525eda_duckdb.md: https://gist.github.com/ilyamusabirov/9491e5ce6ae2fc63d6222609cebd0588
"""

import argparse
from pathlib import Path
import duckdb


# Used Claude in setting up script (args) for command/bash use 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", 
                   default="../data/raw/Electronics.jsonl.gz")
    p.add_argument("--meta", 
                   default= "../data/raw/meta_Electronics.jsonl.gz")
    p.add_argument("--out-dir", 
                   default="../data/processed")
    p.add_argument("--row_limit",   type=int, 
                    default=20000, help="Max rows from each source (default: 20000)")
    p.add_argument("--subset_sample_size",  type=int, 
                   default=500,   
                   help="Unique parent_asin keys in sample (default: 500)")
    p.add_argument("--review-cols", 
                   nargs="+", #to accept multiple columns as input by spaces and then add them as a list to args 
                   default=["title", "helpful_vote"],
                   help="Review columns to include (space-separated)")
    p.add_argument("--meta-cols",
                   nargs="+", #to accept multiple columns as input by spaces and then add them as a list to args 
                   default=["main_category", "title", "features", "description", "details"],
                   help="Meta columns to include (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    output  = Path(args.out_dir)
    reviews_raw = str(output / "reviews_raw.parquet")
    meta_raw    = str(output / "meta_raw.parquet")
    merged_out  = str(output / "merged.parquet")
    sample_out  = str(output / "sample.parquet")

    con = duckdb.connect()

# 1) convert files to parquet (x2 --> meta + reviews) ############################################################

    # Meta
    con.execute(f"""
        COPY (SELECT * FROM read_json_auto('{args.meta}') LIMIT {args.limit})
        TO '{meta_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"meta data exported as a parquet file")

    # Reviews
    con.execute(f"""
        COPY (SELECT * FROM read_json_auto('{args.reviews}') LIMIT {args.limit})
        TO '{reviews_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"reviews data exported as a parquet file")

# 2) merge files on parent_asin key ############################################################

   # need to convert desired columns to one large string for SQL
    raw_cols = ", ".join(f"raw.{col}" for col in args.review_cols)
    meta_cols = ", ".join(f"meta.{col}" for col in args.meta_cols)

    # left join from meta so every product row is preserved (even with no reviews)
    con.execute(f"""
        COPY (
            SELECT meta.parent_asin, {meta_cols}, {raw_cols}
            FROM read_parquet('{meta_raw}') as meta
            LEFT JOIN read_parquet('{reviews_raw}') as raw USING (parent_asin)
        )
        TO '{merged_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    
    print(f"merged meta and reviews exported to {merged_out} as a parquet file")

# 3) create subset ############################################################


    con.close()

if __name__ == "__main__":
    main()