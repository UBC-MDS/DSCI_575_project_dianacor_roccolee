
"""
Usage:
    python convert_parquet.py # to convert raw downloaded jsonl.gz files to parquet, and create a sample subset for quicker iteration
References: 
    ilyamusabirov/525eda_duckdb.md: https://gist.github.com/ilyamusabirov/9491e5ce6ae2fc63d6222609cebd0588
"""

import argparse
from pathlib import Path
import duckdb
from utils import *


default_review_cols = read_review_txt_columns()
default_meta_cols = read_meta_txt_columns()

# Used Claude in setting up script (args) for command/bash use 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", 
                   default="data/raw/Electronics.jsonl.gz")
    p.add_argument("--meta", 
                   default= "data/raw/meta_Electronics.jsonl.gz")
    p.add_argument("--out-dir", 
                   default="data/processed")
    p.add_argument("--subset_sample_size",  type=int, 
                   default=500,   
                   help="Unique parent_asin keys in sample after being merged, & number of rows for each main source file")
    p.add_argument("--review-cols", 
                   nargs="+", #to accept multiple columns as input by spaces and then add them as a list to args 
                   default=default_review_cols,
                   help="Review columns to include (space-separated)")
    p.add_argument("--meta-cols",
                   nargs="+", #to accept multiple columns as input by spaces and then add them as a list to args 
                   default=default_meta_cols,
                   help="Meta columns to include (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    output  = Path(args.out_dir)
    reviews_raw = str(output / "reviews_raw.parquet")
    reviews_subset = str(output / "reviews_raw_subset.parquet")

    meta_raw    = str(output / "meta_raw.parquet")
    meta_subset = str(output / "meta_raw_subset.parquet")

    merged_out  = str(output / "merged.parquet")
    merged_subset  = str(output / "merged_subset.parquet")


    con = duckdb.connect()
    print("[STATUS] Connected to DuckDB to start the conversion process.")

#################### 1) convert files to parquet (x2 --> meta + reviews) #############################

    # Meta
    con.execute(f"""
        COPY (SELECT * FROM read_json_auto('{args.meta}',
                            ignore_errors=true
                            -- sample_size=-1 "forces to scan whole file before inferring types — slow but more accurate"
                            ))
        TO '{meta_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[DONE] Meta data exported as full-sized parquet file.")

    # Reviews
    # con.execute(f"""
    #     COPY (SELECT * FROM read_json_auto('{args.reviews}',
    #                         ignore_errors=true
    #                         -- sample_size=-1 "forces to scan whole file before inferring types — slow but more accurate"
    #                         ))
    #     TO '{reviews_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)
    # """)
    # print(f"[DONE] Reviews data exported as full-sized parquet file.")

    # Reviews
    con.execute(f"""
    COPY (
        WITH ranked_reviews AS (
            SELECT
                parent_asin,
                title,
                helpful_vote,
                ROW_NUMBER() OVER (
                    PARTITION BY parent_asin
                    ORDER BY helpful_vote DESC
                ) AS rn
            FROM read_json_auto(
                '{args.reviews}',
                ignore_errors=true
                -- sample_size=-1  "forces to scan whole file before inferring types — slow but more accurate"
            )
        )
        SELECT
            parent_asin,
            STRING_AGG(title, ' | ' ORDER BY helpful_vote DESC) AS combined_reviews
        FROM ranked_reviews
        WHERE rn <= 10
        GROUP BY parent_asin
    )
    TO '{reviews_raw}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
    print(f"[DONE] Reviews data exported — 1 row per product, top 10 reviews merged.")

################################ 2) merge files on parent_asin key ##########################################

    print(f"[STATUS] Starting merge of both meta and reviews files. This may take a while...")
    # need to convert desired columns to one large string for SQL
    meta_cols = ", ".join(f"meta.{col}" for col in args.meta_cols)
    # reviews parquet only contains parent_asin + combined_reviews after aggregation & can then ignore args.review_cols (just use all cols)

    # left join from meta so every product row is preserved (even with no reviews)
    con.execute(f"""
        COPY (
            SELECT meta.parent_asin, {meta_cols}, raw.combined_reviews
            FROM read_parquet('{meta_raw}') as meta
            LEFT JOIN read_parquet('{reviews_raw}') as raw USING (parent_asin)
        )
        TO '{merged_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(f"[DONE] Merged both meta and reviews and exported.")

########################################## 3) create merged subset ##########################################

# exporting only the first "subset_sample_size" number of rows (via cmd argument)
    # Meta subset
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{meta_raw}')
              LIMIT {args.subset_sample_size})
        TO '{meta_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[DONE] Exported a subset of the full-sized meta file.")

    # Reviews subset
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{reviews_raw}')
              LIMIT {args.subset_sample_size})
        TO '{reviews_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[DONE] Exported a subset of the full-sized review file.")

# exporting rows only belonging to the first "subset_sample_size" number of unique parent_asin keys (via cmd argument)
    con.execute(f"""
        COPY (
            WITH subset_N_products AS (
                SELECT DISTINCT parent_asin
                FROM read_parquet('{merged_out}')
                LIMIT {args.subset_sample_size}
            )
            SELECT meta.* 
            FROM read_parquet('{merged_out}') as meta
            INNER JOIN subset_N_products USING (parent_asin)
        )
        TO '{merged_subset}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"[DONE] Exported a subset of the full-sized merged (meta + reviews) file.")

    con.close()
    print(f"[STATUS] Completed all conversions and merges. Closed DuckDB connection.")


if __name__ == "__main__":
    main()