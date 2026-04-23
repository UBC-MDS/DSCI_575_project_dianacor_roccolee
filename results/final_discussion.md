# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: _10,000_
- Changes to sampling strategy (if any): _None_

### LLM Experiment
- Models compared (name, family, size)
  - qwen/qwen3-32b
  - openai/gpt-oss-120b

- Prompt used (copy it here)
    >"""
    >You are a helpful Amazon shopping assistant specializing in the electronics selection/category.
    >Answer the question using ONLY the following context, which contains product metadata for electronic products and customer reviews.
    >
    >The context includes these fields:
    >- ASIN: Unique product identifier
    >- Product Title: Name of the electronic product
    >- Document: A concatenation of product metadata and reviews information
    >
    >Guidelines:
    >- Always cite the ASIN and product title in your answer, as well as the star rating if possible. 
    >- When summarizing opinions, reference the review rating to support your claim
    >- If the context does not contain enough information to answer, say: "Unfortunately, I don't have enough review data to answer this confidently."
    >- Keep answers concise and focused on the user's specific question
    >- Never recommend a product that is not present in the provided context
    >"""

- Outputs: The summarized & simplified can be found below. The full output to the test queries can be found in `./results/query_results_milestone2.csv`.

| Query | Qwen/Qwen3-32B Response | OpenAI/GPT-OSS-120B Response |
|-------|------------------------|------------------------------|
| wireless bluetooth headphones | Top picks: **B0C68LS4MX** - Wireless Earbuds Bluetooth 5.3 (4.7★): 42hr playtime, IPX7, portable charging case. **B0BNVMXLXQ** - Wireless Bluetooth Earbuds IPX7 (4.5★): Bluetooth 5.0, instant pairing, deep bass. For noise cancellation: **B0CGVVRCQ3** TECKNET Bluetooth Headset (4.0★). Budget option: **B0C5SHB6VX** FORTECLEAR V5.3 (3.8★) with 50hr playtime. | Top picks: **B0C68LS4MX** (4.7★) - praised for stable BT 5.3, HiFi sound, ~42hr playtime, water resistance. **B0BNVMXLXQ** (4.5★) - effortless pairing, IPX7, deep bass. **B0CGVVRCQ3** TECKNET (4.0★) - AI noise cancelling, 49ft range, mute button. **B0C5SHB6VX** FORTECLEAR (3.8★) - 50hr battery, IPX7. **B00KA6SZRO** Energy BT5+ (3.8★) - NFC pairing. **B0BTM2CL52** (3.2★) - neckband, 18hr battery. **B0C39ZM9MT** (3.1★) - BT 5.2, ENC mic. Recommendation: B0C68LS4MX is the top choice. |
| 1080p gaming monitor | **B08X4BGM5P** AUZAI 27" 144Hz 1ms IPS (4.6★): 1080p, G-Sync & FreeSync, HDMI/DisplayPort/USB, frameless design. AOC 22" B009EAEXSC (5.0★) also available but 60Hz less ideal for gaming. AUZAI recommended for competitive play. | **B08X4BGM5P** AUZAI 27" 144Hz 1ms IPS (4.6★): FHD 1920×1080, up to 165Hz, 1ms response, IPS viewing angles, FreeSync/G-Sync, adjustable tilt, VESA mount. Recommended for smooth, tear-free gameplay. |
| high performance work laptop | **B07SW4GYKZ** Lenovo ThinkPad L14 Gen2 (5.0★): Intel Core i5-1135G7 up to 3.6GHz, 20GB DDR4, 256GB SSD. Also consider **B0CK8K4RFL** ACEMAGIC (4.3★): 12th Gen Intel N95, 16GB RAM, 512GB SSD. Avoid HP Pavilion B0BG3KCY4C (1.0★). | **B07SW4GYKZ** Lenovo ThinkPad L14 Gen2 (5.0★): 14" FHD touchscreen, Intel Core i5-1135G7, 20GB RAM, 256GB NVMe SSD, Wi-Fi 6, HDMI, Windows 11 Pro. Top-rated for professional use with robust business-ready features. |
| super fast wireless charger | **B000KB30UQ** ASAITEKE Phone to HDMI Adapter (5.0★): Includes wireless car charger supporting 10W for Samsung Galaxy and 7.5W for iPhone. Features overcharge, temperature, and short-circuit protection. No dedicated standalone wireless chargers found in context. | **B000KB30UQ** ASAITEKE Phone to HDMI Adapter (5.0★): Doubles as wireless fast-charging car mount. Supports 10W for Samsung Galaxy S9/S8/S7/Note 8 and 7.5W for iPhone X/8. Includes safety protections, works with most cases. Perfect 5-star rating from users. |
| white gaming mouse for left handed people | No white left-handed gaming mice found. Closest option: **B08HVGX6QX** RATEL Wireless Mouse (4.8★) — ambidextrous, symmetric design, mint color, power-saving modes, compact and portable. | Insufficient review data to answer confidently. |

- Key observations (found in ./results/query_results_milestone2.csv):

For most test queries, both models performed well in giving coherent answers based on the provided context. Both models followed the guidelines given in the system prompt well and recommended several products that fit the user's query. The main difference between the responses of the models is in the reasoning for recommending certain products over others. The Qwen model seems to prefer spelling out each reason in sentences, while the OpenAI model perfers to lay out its reasoning in tables and bullet points, which are probably easier for the user to read. The difference in the number of parameters started to show in the forth query for "super fast wireless chargers", where the OpenAI model was able to infer more from the adjectives in the query to return multiple products, while the Qwen model was only able to find one product that matched the query before mentioning that the given context didn't include any other products that satisfy the user's query. Finally, for the most complex query of "white gaming mouse for left handed people", both models mentioned that the given context didn't contain a product that matched the query, however the Qwen model offered a product that it thought was a good substitute, while mentioning that the substitute didn't fit the "white" or "gaming" criteria of the query.

- Which model you chose and why:

Given that both models performed decently well, but the Qwen model contains about 1/4 of the number of parameters of the OpenAI mode, we will proceed with the Qwen model as the default in our hybrid pipeline.

## Step 2: Additional Feature (state which option you chose)

### What You Implemented

- Description of the feature
  - Cloud deployment of local web app through the Posit Cloud platform, accessible via this [link](https://roccolee18-amazon-electronics-recommender-app.share.connect.posit.cloud)

- Key results or examples
  - We successfully published our local Shiny web app onto the Posit Cloud platform. Achieving this required creating a fork of our repo and refactoring the file structure to only include the web app related code. The biggest challenge was scanning through our web-app code to find file dependancies and ensure those were transfered/kept in the forked repo to maintain functionality.

## Step 3: Improve Documentation and Code Quality

### Documentation Update
- update script name references to match new names
- add instructions for the groq api key
- mentioned increased sample size + a disclaimer for re-running pre-final submission scripts (may no longer match the outputs mentioned in the `____-discussion.md`
- mention the new connect posit cloud website as the feature implemented and the online-version of the local app

### Code Quality Changes
- refactored 1 script into each of it's sub-steps. Original function was convert_parquet and it refactored into scripts 1-3
- moved away from local llm model to online gorq one for speed and quality 
- removed all hardcoded paths and used parse.args for all scripts
- moved all functions into utils module and added docstrings to all functions
- reduced the app.py CSS into it's own file and fix connections of retreivers from refactored changed
- changed the UI appearance of the app to a nicer and more readable light version

## Step 4: Cloud Deployment Plan
- Data Storage
Due to the large size of our database, we created helper functions that convert the raw metadata and reviews into parquet files which are much smaller and easier store. Because of this optimization, we believe we can store the parquet files on AWS S3.

The raw data itself can also live on an AWS S3 bucket, as they won't be accessed frequently during processing, and S3 has scalable storage to meet the space needed for the larger, raw JSONL files.

Since we've only tested our app with 1 or 2 users at a time, loading the indexes (both semantic vectors and BM25) onto memory worked well enough. If we are expecting a higher volume of traffic once our app gets deployed, we may consider also bringing the indexes onto a cloud service like AWS OpenSearch to allow mulitple app instances to query the same index without requiring each device to load it into memory every time a new session is started

- Compute
Regarding the computation of our programs, due to the large number of specific requirements needed, a containerizer like Docker would help improve usability and ensure package versions don't conflict, which is an issue we briefly ran into. The container can then be hosted on cloud services like AWS ECS. The largest advantage to using this service is its elastic billing system and integration with the AWS ecosystem that is used in other parts of our project

A potential limitation of using Shiny for Python to build our app is that having multiple users may block each other during larger RAG queries. In those instances, our options are either to run mulitple app instances, which may require more compute resources and redundant loading, or offloading the LLM inference onto another cloud service like AWS SQS so the app can still be responsive while inference is handled in the background.

In terms of handling the LLM inference itself, that depends on the size of model we choose as our final one. In our Milestones, we experimented with lighter Qwen and OpenAI models, so that is what we will assume the full deployed app will also use. Hosting those models will require a GPU instance so users aren't stuck waiting a long time for a single query, however that is an expensive option to keep running all the time. Therefore, we believe using the HuggingFace API or AWS Bedrock is a better alternative, as the payment scheme is only per token rather than per hour. It also offloads the responsibility of managing GPU infastructure as well.

- Streaming/Updates
To continually process new data, a scheduled job on the cloud platform (for example, AWS Lambda) will be a cost-efficient way to monitor the S3 bucket for new data drops and trigger the appropriate scripts to process the new data and create new indexes in batches for the RAG to reference. The data drops (new reviews and product metadata) can be obtained through a scheduled batch download from the Amazon dataset.

Keeping the pipeline up to date can be done by building a CI/CD pipeline with Github Actions to automatically redeploy the app when new code is pushed. As mentioned above, the indexes can be updated with new product information by having a AWS Lambda function monitor for new data drops in the S3 bucket and trigger a run of the pre-processing pipeline to get the updated index into the RAG pipeline.
