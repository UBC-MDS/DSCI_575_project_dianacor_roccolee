# DSCI_575_project_dianacor_roccolee

## Description of Dataset
This project uses a data set of Amazon reviews and product metadata grouped by product category. Specifically, we chose to focus our data set on products under the Electronics category. 
The data sources can be found below:

Dataset Website: https://amazon-reviews-2023.github.io/
Hugging Face: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## Recreating Project Workflow

1. Clone the repo into the desired folder using this command in a new terminal window:
```bash
git clone git@github.com:UBC-MDS/DSCI_575_project_dianacor_roccolee.git
```

2. Make and activate the enviroment using the command below in the same terminal: 
```bash
conda env create -f environment.yml 
```

3. Download the data set by running (assuming you are in the root project directory) 
```bash 
python ./src/direct_datadownload.py
``` 
or alternatively:
1. Manually download the data from [here](https://amazon-reviews-2023.github.io/)
- Download both the reviews and meta of the *Electronics* category
2. Move zip files to the data/raw directory
3. Extract (unzip) files in the same directory
4. Run bellow code to convert from .jsonl / .json.gz to parquet
```bash
python src/convert_parquet.py \
  --reviews data/raw/Electronics.jsonl.gz \
  --meta data/raw/meta_Electronics.jsonl.gz \
  --subset_sample_size 500 
```
- By default, the parquet file output will be saved in `./data/processed`

4. Create documents from the data set using:
```bash
python create_documents.py
```

5. To use BM25 search, run the `bm25.py` file with the appropriate arguments, or just call it plainly to use default arguments
```bash
python ./src/bm25.py --query "sony headphones"
```

6. To use semantic search, run the `semantic.py` fule with the appropriate arguments, or just call it plainly to use default arguments
```bash
python ./src/semantic.py --query "sony headphones"
```

7. Alternatively, experiment with the search through a web app by running:
```bash
shiny run ./app/query_app.py
```