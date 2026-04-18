## LLM Model Choice Rationale

For building RAG functionality in our project, we decided to use the Qwen/Qwen2.5-1.5B model. We chose this model because of our relatively limited local computing power and its reliable performance in basic tasks such as retrieving product recommendations. There was an even lighter version of this model (Qwen/Qwen2.5-0.5B) which we decided not to go for because the performance tradeoff would be too high. We've done some research on model performance and found that models of the 0.5B category would struggle to gather information from multiple documents, which is what we plan to do with our RAG model. Further, The chosen model being a decoder-only model makes it well suited for the task of a RAG. Some alternative decoder-only models we could have chosen include: microsoft/phi-2, microsoft/phi-3/mini-4k-instruct or google/gemma-2-2b-it

## Manual / Qualitative Evaluation for Hybrid RAG Workflow

| # | Query | Response |
|---|-------|---------------|
| 1 | wireless bluetooth headphones | "Based on the average ratings provided, wireless Bluetooth headphones with a rating of 3.8 is a suitable option for your needs. The higher average rating suggests that these headphones have received positive feedback from users, indicating that they meet the expectations set by consumers. However, it's always important to consider other factors such as comfort, battery life, and compatibility before making a final decision." |
| 2 | 1080p gaming monitor | "Based on the provided information, the product that best fits your requirement for a 1080p gaming monitor is the "GIGABYTE GeForce RTX 3060 Gaming OC 12G (REV2.0) Graphics Card, 3X WINDFORCE Fans, 12GB 192-bit GDDR6" with an average rating of 4.7. This monitor is specifically designed for high-performance gaming and offers excellent graphics capabilities suitable for modern gaming needs." |
| 3 | high performance work laptop | "Based on the provided context, I would recommend the product with the ASIN B00BN0LHF8, titled ""Samsung NP540U3C-A03UB Series 5 13.3"" Touch-Screen Laptop. This product has an average rating of 3.5, indicating that it has a reasonable level of performance for its intended purpose, which is a high performance work laptop." |
| 4 | super fast wireless charger | "I'm sorry, but I'm unable to provide a meaningful answer based on the provided context. The context only contains information about product ratings but does not mention any super fast wireless charger. Please provide additional context or clarify your question." |
| 5 | extension cord for usb-a to usb-c | "The most suitable product for an extension cord to connect USB-A to USB-C is the one with the ASIN B08XB3B6FY. This product has an average rating of 4.5 stars, indicating a higher rating than the other options, suggesting it is more satisfactory in terms of its performance and customer satisfaction." |

Query 1: wireless bluetooth headphones

| Dimension | What to assess | Yes/No |
|---|-------|-----|
| Accuracy | Is the answer factually correct based on the reviews? | No |
| Completeness | Does the answer address all aspects of the question? | No |
| Fluency | Is the answer natural, clear, and easy to read? | Yes |

Query 2: 1080p gaming monitor

| Dimension | What to assess | Yes/No |
|---|-------|-----|
| Accuracy | Is the answer factually correct based on the reviews? | Yes |
| Completeness | Does the answer address all aspects of the question? | Yes |
| Fluency | Is the answer natural, clear, and easy to read? | Yes |

Query 3: high performance work laptop

| Dimension | What to assess | Yes/No |
|---|-------|-----|
| Accuracy | Is the answer factually correct based on the reviews? | Yes |
| Completeness | Does the answer address all aspects of the question? | Yes |
| Fluency | Is the answer natural, clear, and easy to read? | Yes |

Query 4: super fast wireless charger

| Dimension | What to assess | Yes/No |
|---|-------|-----|
| Accuracy | Is the answer factually correct based on the reviews? | No |
| Completeness | Does the answer address all aspects of the question? | No |
| Fluency | Is the answer natural, clear, and easy to read? | Yes |

Query 5: extension cord for usb-a to usb-c

| Dimension | What to assess | Yes/No |
|---|-------|-----|
| Accuracy | Is the answer factually correct based on the reviews? | Yes |
| Completeness | Does the answer address all aspects of the question? | No |
| Fluency | Is the answer natural, clear, and easy to read? | No |


- Overall, introducing a RAG pipeline to our inital semantic/BM25 search methods resulted in clearer and more customizable responses to user queries. Through our testing, we discovered that the initial system prompt plays a big role in determining the performance of the RAG. An example of this is the product title not being mentioned in the response because we didn't explicitly tell the LLM to include the product title in its response.
- One of the limitations of our Hybrid RAG workflow is that it is subject to weight assigned to the BM25 and Semantic retrievers. Currently, it is favored toward BM25 which might not be the most optimal, as it is unlikely the user will include exact phrasing of reviews or product titles in their queries. Further, the quality of the output is heavily subject to the HuggingFace model chosen, as more expensive and capable models will have larger context windows and be able to make smarter inferences.
- Improvement that could be further made to this workflow include: a randomized search on the weights of the BM25 and semantic retrievers to find the ratio that gives the best results, as well as experimenting on the size of chunking to allow for more varied results while still keeping a good amount of information per document.