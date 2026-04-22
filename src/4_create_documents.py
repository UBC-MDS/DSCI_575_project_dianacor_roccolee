import argparse
from pathlib import Path
import duckdb

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
    p = argparse.ArgumentParser()
    p.add_argument("--input",   
                   default="data/processed/merged_subset.parquet")
    p.add_argument("--out-dir", 
                   default="data/processed/product_documents.parquet")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out_dir)

    # foundation of query supplied by Claude 
    # concats title, main_category, and features (after expanding the array)
    # especially for the .as_posix() syntax for paths to work with duckdb window/copy command 
    # and the MAP_KEYS lines to handle the strange mapping structure of the details column 
    QUERY = f"""
    SELECT
        parent_asin,
        LEFT(ANY_VALUE(title), 100) AS product_title,
        ANY_VALUE(average_rating) AS average_rating,
        CONCAT_WS(' ',
            ANY_VALUE(main_category),
            ANY_VALUE(title),
            ARRAY_TO_STRING(ANY_VALUE(features), ' '),
            ARRAY_TO_STRING(ANY_VALUE(description), ' '),
            ARRAY_TO_STRING(MAP_KEYS(ANY_VALUE(details)), ' ') || ' ' || ARRAY_TO_STRING(MAP_VALUES(ANY_VALUE(details)), ' '),
            ANY_VALUE(combined_reviews)
        ) AS document
    FROM read_parquet('{input_path.as_posix()}')
    WHERE parent_asin IS NOT NULL
    GROUP BY parent_asin
    """

    con = duckdb.connect()
    print("[STATUS] Connected to DuckDB to start document creation.")
    con.execute(
        f"COPY ({QUERY}) TO '{output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)",
    )
    print("[STATUS] Documentation creation completed.")
    print("[NOTE] Currently product documentation is just from the merged subset\n     (sample size according to arguments passed when created).")
