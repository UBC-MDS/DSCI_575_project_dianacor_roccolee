import argparse
from pathlib import Path
from utils import * 
# pip install requests tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url",  
                   default="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw", 
                   help="Download base URL source")
    p.add_argument("--out-dir",  
                   default="data/raw", 
                   help="Directory to save downloaded files.")
    p.add_argument("--subset",   
                   default="Electronics", 
                   help="Category subset to download (e.g. Electronics, Books).")
    p.add_argument("--meta",     
                   action=argparse.BooleanOptionalAction,  #Mentioned by Clause to properly parse True/False flag for downloading 
                   default=True, 
                   help="Download the meta file.")
    p.add_argument("--reviews",  
                   action=argparse.BooleanOptionalAction, 
                   default=True, 
                   help="Download the reviews file.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # in case someone just deletes the whole folder

    files_map = file_name_source_map(args.base_url,args.subset, args.meta, args.reviews)
    for filename, specific_url in files_map.items():
        print(f"Request to download: {filename}")
        download_request(specific_url, out_dir, filename)
    print("All raw files downloaded")