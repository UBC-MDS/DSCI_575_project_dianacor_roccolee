import argparse
from pathlib import Path
from datasets import load_dataset




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", 
                   default="data/raw")
    p.add_argument("--meta",
                   default=True,   
                   help="Whether to download 'meta' source file. Default True")
    p.add_argument("--reviews", 
                   default=True,   
                   help="Whether to download 'meta' source file. Default True")
    return p.parse_args()



def main():
    args = parse_args()
    output  = Path(args.out_dir)

    meta_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics", 
                       split="full", 
                       trust_remote_code=True)

    review_data = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Electronics", 
                        trust_remote_code=True)