import argparse
from pathlib import Path
from utils import file_name_source_map, download_request

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
        print(f"[STATUS] Request to download: {filename}")
        download_request(specific_url, out_dir, filename)
        print(f"[SAVED]")
    print("[DONE] All raw files downloaded")