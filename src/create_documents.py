import argparse
from pathlib import Path
import duckdb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   
                   default="data/processed/merged_subset.parquet")
    p.add_argument("--output", 
                   default="data/processed/product_documents.parquet")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # currently just concats title, main_category, and features (after expanding the array)
    # foundation of query supplied by Claude - including the .as_posix() syntax for paths to work with duckdb window/copy command
    QUERY = f"""
    SELECT
        parent_asin,
        LEFT(ANY_VALUE(title), 100) AS product_title, -- only first 100 characters of title(they are quite long)
        CONCAT_WS(' ',
            ANY_VALUE(main_category), -- fine b/c identical across all duplicated rows
            ANY_VALUE(title),
            ARRAY_TO_STRING(ANY_VALUE(features), ' '), 
            ARRAY_TO_STRING(ANY_VALUE(description),' '),
            ARRAY_TO_STRING(MAP_KEYS(ANY_VALUE(details)), ' ') || ' ' || ARRAY_TO_STRING(MAP_VALUES(ANY_VALUE(details)), ' ') -- flattening the details which is a MAP(VARCHAR, JSON) into a single string of keys and values
        ) AS document
    FROM read_parquet('{input_path.as_posix()}')
    WHERE parent_asin IS NOT NULL
    GROUP BY parent_asin
    """

    con = duckdb.connect()
    con.execute(
        f"COPY ({QUERY}) TO '{output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)",
    )

if __name__ == "__main__":
    main()