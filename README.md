# DSCI_575_project_dianacor_roccolee

make and activate enviroment via command (in root repo terminal): 
```
conda  env create -f environment.yml 
```

Data sources:

Dataset Website: https://amazon-reviews-2023.github.io/
Hugging Face: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Data set up process:

both the reviews and meta - SPECIFICALLY THE "ELECTRONICS" ONE

download data via: python `src/direct_datadownload.py` # this might take a very long time as it's bottelnecked by the server throughput. Could be faster to just manually download via website

or alternatively/manually
1. manually download the data from the website (link), both the reviews and meta - SPECIFICALLY THE "ELECTRONICS" ONE
2. move zip files to the data/raw directory
3. export files in the same directory
4. run bellow code to convert from .jsonl / .json.gz to parquet


Then convert files to parquet 
```
python src/convert_parquet.py \
  --reviews data/raw/Electronics.jsonl.gz \
  --meta data/raw/meta_Electronics.jsonl.gz \
  --subset_sample_size 500 
  # add the col extra arguments
```

Create documents via: 