
############################## For convert_parquet.py and create_documents.py script ##############################

# helper function from Claude to keep the columns being used in other scripts consistent 
# and easily editable in one place via a text file where milestone exploration notebook is
# (instead of hardcoding them in multiple scripts which could change) 

    #specific with default meta columns path 
def read_meta_txt_columns(filepath = "notebooks/meta_columns.txt"):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

    #specific with default review columns path 
def read_review_txt_columns(filepath = "notebooks/review_columns.txt"):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]
    

############################## For huggingface_datadownload.py script ##############################
import requests
from tqdm import tqdm
from pathlib import Path

def file_name_source_map(base_url, subset, meta, reviews):
    files = {}
    if reviews:
        files[f"{subset}.jsonl.gz"] = f"{base_url}/review_categories/{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    if meta:
        files[f"meta_{subset}.jsonl.gz"] = f"{base_url}/meta_categories/meta_{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    return files


def download_request(specific_url, output, filename):
    fullpath = Path(output) / filename

    if fullpath.exists(): # prevent from a taxing re-download
        print(f"Already exists, skipped: {filename}")
        return

    print(f"Downloading: {filename}")
    request = requests.get(specific_url, stream=True) #
    request.raise_for_status()

    # Following code below that handles request-downloads is from Claude as it's a nice-to-have and out of scope of assignment
    total = int(request.headers.get("content-length", 0))
    with open(fullpath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=fullpath.name
    ) as bar:
        for chunk in request.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved: {filename} in {output}")

############################## Query using semantic search ##############################
def semantic_search(docs, model, index, query, k=5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'parent_asin': docs.iloc[idx]['parent_asin'],
            'product_title': docs.iloc[idx]['product_title'],
            'distance': dist
        })
    return results