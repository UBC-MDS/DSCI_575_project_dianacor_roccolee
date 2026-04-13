# Amazon Electronics Review Search

This repo uses data of Amazon product & their reviews (specifically for the Electronics category) to compare 2 different retrievals systems: BM25 (keyword-based) and Semantic (embedding-based), to search and compare product results via user queries.

## Description of Dataset

This project uses the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset, focusing on the **Electronics** category. This can also be found in [Hugging Face Website]( https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-202). It combines two data sources:

- **Reviews** (`Electronics.jsonl.gz`): dataset of individual user reviews with fields including `rating`, `text`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, and `verified_purchase`
- **Metadata** (`meta_Electronics.jsonl.gz`): dataset of product-level information including `title`, `price`, `average_rating`, `main_category`, and `store`

These two sources are joined on `parent_asin` to produce a merged dataset (reproducing the work flow instructions below). 

### Preprocessing:

Via EDA done in `milestone_exploration.ipynb`, some columns were dropped or converted in a more palatable way for document creation/pipeline requirements. More explanation can be found there. 

In short: the following columns were dropped in meta `["images", "videos", "subtitle", "author", "bought_together", "rating_number", "average_rating", "price", "store"]` and only `["title", "helpful_vote", "parent_asin"]` was retained in reviews. The dropped columns either do not work well in the retrieval pipeline without serious extra steps (images/videos) or would increase the document size without meaningful additions. The second reason (lack of meaningful additions) is also the main reason why so many of the review columns were dropped - as the document size would explode - but only add noise.

## Current Types of Retrieval Systems:

Currently there are 2 types of retrieval systems being explored:
1. BM25 -> Keyword TF-IDF based. Expected to preform the best when there's high exact-word matching between a query and the products.
2. Semantic -> Embedding-based. Expected to preform the best when there's a more natural language or conceptual queries and the products semantic best match the overall meaning and intent of the user query.capturing meaning beyond exact keyword overlap.

Comming soon:
- Hybrid of BM25 and Semantic
- RAG

## Recreating Project Workflow

> NOTE: If files in `data/processed` are removed -> Steps 1-4 must be done first before running the milestone1_exploration (requires parquet format).

> NOTE: If short on time: steps 3 & 4 (which are very time consuming) can be skipped. This works since subset versions of data sources (already converted) are already exported to the repo - and the pipeline currently just uses a subset due to time constraints.

### 1. Clone the Repository
Clone the repo into the desired folder using this command in a new terminal window:
```bash
git clone git@github.com:UBC-MDS/DSCI_575_project_dianacor_roccolee.git
cd DSCI_575_project_dianacor_roccolee
```

### 2. Create and Activate the Environment
Make and activate the environment using the command below in the same terminal (at the root of the repo): 
```bash
conda env create -f environment.yml
conda activate amazon-recommender # or whatever the custom env name might be
```

### 3. Download the Dataset
> Note: Downloads are very large. Expect 45–60+ minutes depending on your connection. The automated method may be even slower due to server-side rate limits.

**Option A — Manual download  (recommended):**
1. Go to the [dataset website](https://amazon-reviews-2023.github.io/).
2. Locate the *Electronics* category.
3. Download both the reviews and metadata files via the clickable links.
4. Move the zip files to `data/raw/`
5. Extract them in place

**Option B — Automated download**:
```bash
# Via terminal in the root project directory 
python ./src/direct_datadownload.py 
```

### 4. Convert to Parquet
Run bellow code to convert from .jsonl / .json.gz to parquet:
> This step might also take quite long due to the large files conversion and merging the two. Estimated to be ~10-15 minutes.

```bash
python src/convert_parquet.py \
  --reviews data/raw/Electronics.jsonl.gz \
  --meta data/raw/meta_Electronics.jsonl.gz \
  --subset_sample_size 500
```
### 5. Create Search Documents
This prepares the processed data as document objects used by the retrieval systems.

```bash
python ./src/create_documents.py
```

### 6. Run Retrievals

This creates and exports the required embeddings and index's for both retrieval methods. Both scripts accept a `--query` argument to test out a custom query for each respective method (ex: `<...>.py --query "sony headphones"`) - but this is a optional feature and not required.

```bash
python ./src/bm25.py  # BM25 (keyword-based) search
python ./src/semantic.py # Semantic search
```

## 7. Run Retrieval on Example Queries
This runs examples queries that are available in `results/queries.csv` (this can be changed an customized if desired) against both retrieval methods and outputs the results in `results/query_results.csv`. From these 10 example queries provided 5 were chosen to compare, reflect and review the performance of the methods.  

```bash
python src/query_retrieval.py
```

### Run the Web App (*Still Under Construction)

You can also experiment with query search's through a web app by running:
(Note: this is still currently under construction: deferred to Milestone 2 - but some functionality available )

```bash
shiny run ./app/app.py
```

Then open the URL shown in your terminal. Use the different radio buttons to toggle between the different retrieval methods.
