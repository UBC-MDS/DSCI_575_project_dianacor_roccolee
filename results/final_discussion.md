# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used - 10,000
- Changes to sampling strategy (if any) - None

### LLM Experiment
- Models compared (name, family, size)
- Qwen/Qwen2.5-0.5B 
- microsoft/Phi-3-mini-4k-instruct

- Results and discussions
    - Prompt used (copy it here)
    - You are a helpful Amazon shopping assistant.
    Answer the question using ONLY the following context (which contains real product reviews + metadata). 
    Include the product's average rating as part of your reasoning for selecting a certain product, the higher the rating the better the product.
    Always cite the product ASIN when possible. If the answer isn't in the context, say so.

    - Results (found in ./results/final_query_results_v2.csv)
    - For most of the Microsoft model's responses, it seems like the responses are clearer and more relevant to the user's queries. Unlike Qwen, the Microsoft model never returned an empty response due to a lack of context. It performed especially will with the "1080p gaming monitor" query, where it successfully identified not only the highest rated product, but also gave rationale specific to that product's features as to why it would be ideal for 1080p gaming. However, it seemed to struggle with the last and most complex query of "white gaming mouse for left-handed people". At first, it recommeneded a product that was completely unrelated, but caught and corrected itself mid-response to a mouse that made more sense for the given query. With a more powerful model like this one, the bottleneck comes from compute and crafting the appropriate prompt for the model to make use of. Because both our computers don't have compatable GPU's for LLM models, we struggled with long wait times for responses to be returned. Given more time, we would certianly have continued experimenting with prompts and post-processing techniques to not only better isolate the model's response, but also put the appropriate constraints on the model through the system prompt to guide it toward better responses.

- Which model you chose and why
- We chose to experiment with the microsoft/Phi-3-mini-4k-instruct model because it can be used with our existing code, meaning no extensive code refactoring is needed for further testing. This model is also known follow instructions and better reasoning given its relatively smaller size.

## Step 2: Additional Feature (state which option you chose)

### What You Implemented

- Description of the feature
- Key results or examples
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update
- Summary of `README` improvements

### Code Quality Changes
- Summary of cleanups

## Step 4: Cloud Deployment Plan
(See Step 4 above for required subsections)