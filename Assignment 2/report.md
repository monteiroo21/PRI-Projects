## Project Description: Part 2 — Semantic Search & AI Integration

Building upon the classical term-matching (BM25) system developed in Assignment 1, this second phase transitions the project from a purely lexical search engine to a **Semantic Search System**. By integrating Neural Networks and Large Language Models (LLMs), the system now understands context and intent, significantly improving retrieval precision and user experience.

The architecture has been upgraded with the following **AI-driven components**:

1.  **Neural Reranker (`CrossEncoder`)**
    To improve the ranking quality of the initial BM25 results, we implemented a **two-stage retrieval pipeline**.
    *   **Retrieve:** The system first fetches a candidate pool (e.g., Top-50) using the existing fast BM25 index.
    *   **Rerank:** These candidates are re-scored using a pre-trained Cross-Encoder model, **`unicamp-dl/mMiniLM-L6-v2-pt-v2`**. This model, fine-tuned for Portuguese, inputs the query and document pairs simultaneously to calculate a semantic similarity score, effectively bubbling the most relevant documents to the top.

2.  **Semantic Snippet Extraction**
    Instead of displaying the first few lines of a document, the system now uses the Neural Reranker to analyze specific paragraphs within the retrieved documents. It identifies and extracts the single most relevant text chunk to display as a snippet, ensuring the user sees the answer immediately in the search results.

3.  **Answer Generation (RAG)**
    We implemented a **Retrieval-Augmented Generation (RAG)** system using Google's **Gemini 2.5 Flash** model.
    *   **Design Choice (Context Window):** Although the assignment suggested using the Top-1 document, we chose to inject the **Top-5 reranked documents** into the LLM context. This design decision mitigates the "Single Point of Failure" risk (where the top document is relevant but lacks the specific answer) and allows the model to synthesize complementary information from multiple sources, resulting in a richer and more robust answer.

4.  **AI-Powered Metadata (Tagging)**
    As an additional enhancement, the system utilizes the LLM to automatically analyze document content and generate structured metadata, including a **Main Category** and **5 Relevant Tags**. This facilitates better content organization without manual labeling.

---

## How to Run the Project

To execute the full information retrieval pipeline — including the new AI capabilities — follow the steps below.

**Prerequisites:**
Ensure you have a `.env` file located in the root of the `Assignment 2/` folder containing your Google API Key for the RAG functionality:

`Assignment 2/.env`:
```env
GOOGLE_API_KEY=your_api_key_here
```

1.  **Clean and preprocess the dataset**
    *(Standard processing from Part 1)*
    ```bash
    uv run python -m wifear.reader
    ```
    This generates `data/ptwiki_clean.jsonl` and `data/docstore.jsonl`.

2.  **Build the inverted index**
    *(Standard SPIMI indexing from Part 1)*
    ```bash
    uv run python -m wifear.entrypoints.cli data/ptwiki_clean.jsonl
    ```
    Creates the positional inverted index `data/index_final.jsonl`.

3.  **Load the index into SQLite**
    ```bash
    uv run python -m wifear.core.load_db
    ```
    Imports the index into `index.db`. This database is now used by the Neural Search Engine to fetch initial candidates before reranking.

4.  **Start the AI Search Interface**
    ```bash
    uv run uvicorn wifear.entrypoints.asgi:app --reload
    ```
    Launches the FastAPI application.
    *   **Neural Reranking** and **RAG** will be automatically enabled if the `GOOGLE_API_KEY` and the `sentence-transformers` models are loaded successfully.
    *   Access locally at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**


---

# 1. Neural Reranking

To overcome the limitations of lexical search (BM25)—which struggles with synonyms, polysemy, and lack of semantic understanding—we implemented a **Neural Reranking** stage. This component acts as a refinement layer that re-scores the initial candidates retrieved by the inverted index based on their actual semantic relevance to the query.

### 1.1 Model Choice: Cross-Encoder
We utilized the **`unicamp-dl/mMiniLM-L6-v2-pt-v2`** model via the `sentence-transformers` library.

*   **Architecture:** This is a **Cross-Encoder** model. Unlike Bi-Encoders (which compute cosine similarity between separate vector embeddings), a Cross-Encoder processes the query and the document **simultaneously** as a single input pair. This allows the model to perform deep self-attention between the query terms and document terms, resulting in significantly higher accuracy for ranking tasks.
*   **Justification:** We selected this specific model because:
    1.  It is **fine-tuned for Portuguese**, ensuring high performance on the PT-Wiki corpus.
    2.  It is based on `mMiniLM-L6`, a distilled model with only 6 layers. This provides an optimal balance between **inference speed** (crucial for a real-time search engine) and **ranking precision**.

### 1.2 Implementation Strategy
The `neural_search` method in `searcher.py` implements the reranking pipeline as follows:

1.  **Candidate Retrieval:** The system first executes a BM25 query to fetch a broad pool of candidates (default `candidates_k=50`).
2.  **Handling Long Documents (Chunking):** Since Transformer models have a token limit (typically 512 tokens), we cannot pass entire Wikipedia articles at once. We implemented a sliding window strategy (`_split_into_token_chunks`) that breaks documents into chunks of 480 terms with an overlap of 64 terms.
3.  **Pairwise Scoring:** We construct pairs of `[Query, Text Chunk]` and pass them to the Cross-Encoder.
    ```python
    scores = self.reranker.predict(pairs, batch_size=32)
    ```
4.  **Max-Score Aggregation:** To determine the final score of a document, we use a **Max-Pooling** strategy. The score of a document is equal to the score of its most relevant chunk. This ensures that if a specific section of a long article answers the query, the document is ranked highly, even if the rest of the text is unrelated.
5.  **Dynamic Snippet Extraction:** As an additional enhancement, we reused the Cross-Encoder to generate semantic snippets. The system splits the document into paragraphs and scores them against the query. The paragraph with the highest semantic score is returned as the snippet, ensuring the user sees the most relevant context immediately.

### 1.3 Evaluated Models

The following models were evaluated under the same pipeline and execution conditions:

- `cross-encoder/ms-marco-MiniLM-L-4-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `cross-encoder/stsb-distilroberta-base`
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `unicamp-dl/mMiniLM-L6-v2-pt-v2`

### 1.4 Evaluation Criteria

Models were compared not only based on the average execution time per query, but also on the reranking quality. This evaluation was based, not only on the contents of the retrieved documents, but also on the quality and on the capacity of the LLM to return an answer, based on the reranked documents, provenient from these models.


### 1.5 Execution Time

The table below reports the **average execution time per query**, considering only the reranking stage (cross-encoder) and a fixed number of candidates per query.

| Model | Query A | Query B | Query C |
|------|------------------------|-----------|---------|
| ms-marco MiniLM L-4 | 4 s | 4 s | 4 s |
| ms-marco MiniLM L-6 | 6 s | 6 s | 6 s |
| ms-marco MiniLM L-12 | 6 s | 6 s | 6 s |
| STSB DistilRoBERTa | 19 s | 18 s | 19 s |
| mMARCO mMiniLM L-12 | 12 s | 12 s | 12 s |
| **unicamp mMiniLM L-6 (PT)** | **6 s** | **6 s** | **6 s** |
---

# 2. Answer Generation (RAG)

The final component of our system transforms the search engine into a Question Answering (QA) system using **Retrieval-Augmented Generation (RAG)**. Instead of simply returning a list of links, the system synthesizes a natural language answer based on the retrieved content.

### 2.1 Implementation Strategy: Context & Prompting

To ensure high-quality responses, we implemented specific design choices regarding context window construction and prompt engineering.

**A. Multi-Document Context (The "Top-5" Strategy)**
A critical design choice in our `generate_answer` method was the context window size. While the assignment suggested using the Top-1 document, we implemented a **Multi-Document Context (Top-5)** strategy.
*   **Rationale:** Relying solely on the Top-1 document introduces a "Single Point of Failure." If the highest-ranked document is semantically similar but lacks the specific fact requested, the LLM cannot answer. By feeding the **Top-5 reranked documents**, we provide the LLM with a broader knowledge base, allowing it to synthesize complementary information from multiple sources and handle complex queries where the answer might be split across different articles.

**B. Prompt Engineering**
We engineered a robust system prompt to ensure the generated answers are accurate, grounded, and linguistically correct. The prompt structure implemented in the code includes:
1.  **Persona Definition:** *"És um assistente inteligente de recuperação de informação."*
2.  **Strict Grounding:** *"A tua tarefa é sintetizar uma resposta... baseada APENAS nestes documentos."* This instruction is crucial to prevent "hallucinations" (inventing facts not present in the source).
3.  **Language Constraint:** *"Responde em Português de Portugal."*
4.  **Fallback Mechanism:** We explicitly instructed the model to fail gracefully: *"Se a resposta não estiver nos documentos, diz 'A informação recuperada não contém a resposta'."*

### 2.2 Model Selection Analysis: Why Gemini 2.5 Flash **Lite**?

We integrated Google's **Gemini 2.5 Flash Lite** model (`gemini-2.5-flash-lite`) via the `google.generativeai` SDK. This decision resulted from a comprehensive analysis comparing Capacity, Rate Limits, and Latency against the standard "Flash" and the heavier "Pro" variants.

#### 2.2.1 Token Limits & Capacity: Parity with Pro Models
A common misconception is that "Lite" models significantly reduce the context window size compared to "Pro" models. However, our technical analysis confirms that **Gemini 2.5 Flash Lite maintains parity with the most powerful models regarding Input Token limits.**

| Model Variant | **Input Token Limit** (Context Window) | **Output Token Limit** | **RAG Suitability** |
| :--- | :--- | :--- | :--- |
| **Gemini 2.5 Flash Lite** | **1,048,576** | **65,536** | **Optimal.** Combines massive context with extreme speed. |
| Gemini 2.5 / 3 Pro | 1,048,576 | 65,536 | **Overkill.** Same context capacity but significantly higher cost/latency. |
| Gemini 2.0 Flash (Legacy) | 1,048,576 | 8,192 | **Limited.** Lower output limit restricts detailed answers. |

**Key Finding:** The "Lite" designation refers to parameter count and cost, **not context capacity**. With a 1 Million token window, the Lite model can easily process our "Top-5" (or even Top-50) documents without truncation, matching the capabilities of the Pro versions but with greater efficiency.

#### 2.2.2 Operational Constraints (RPM)
A decisive factor for a search engine's usability is the API Rate Limits (Requests Per Minute - RPM). As indicated by the official usage metrics (refer to Figure X), the Lite version provides a distinct throughput advantage.

| Model Variant | **RPM Limit** (Requests Per Minute) | **System Impact** |
| :--- | :--- | :--- |
| **Gemini 2.5 Flash Lite** | **10 RPM** | **High Availability.** Doubles the capacity for concurrent queries. |
| Gemini 2.5 Flash | 5 RPM | **Moderate.** Risk of `429 Too Many Requests` errors under load. |
| Pro Variants | ~2-5 RPM | **Bottleneck.** Unsuitable for rapid search iterations. |

As evidenced by the console data, **Gemini 2.5 Flash Lite** offers **double the RPM capacity (10 vs 5)** of the standard Flash model, making it the most robust choice for handling multiple consecutive user queries.

#### 2.2.3 Latency and Performance Benchmarks
Beyond capacity and limits, we conducted internal benchmarks to measure the user's wait time (Latency). Speed is critical for search engine UX.

*   **Gemini 2.5 Flash (Standard):** Average response time of approximately **25 seconds**.
*   **Gemini 2.5 Flash Lite:** Average response time of approximately **13 seconds**.

The **Lite version is nearly 2x faster** in generating the final answer. Waiting 25 seconds for a summary disrupts the user flow, whereas the 13-second benchmark of the Lite version falls within a much more acceptable range for real-time information retrieval.

#### 2.2.4 Quality Comparison & Conclusion
Finally, we analyzed the semantic quality of the outputs. Our testing revealed that for the specific task of RAG—summarizing facts provided in a clear context window—the responses generated by the **Lite** model were qualitatively indistinguishable from those of the standard **Flash** or even **Pro** models.

**Conclusion:**
Since `gemini-2.5-flash-lite` delivers the **same context capacity (1M tokens)** and **comparable response quality**, while offering **half the latency (13s vs 25s)** and **double the throughput (10 RPM vs 5 RPM)**, it is unequivocally the optimal choice for our system.

---

# 3. Further AI Enhancement A: Semantic Snippet Extraction

To achieve the highest grade criteria regarding **AI Enhancements**, we moved beyond static document descriptions. Standard search engines often display the first few lines of a document, which in encyclopedic articles (like Wikipedia) are often generic introductions or navigational metadata that do not explain *why* the document is relevant to the specific user query.

To solve this, we implemented a **Dynamic Semantic Snippet** mechanism using the Neural Reranker.

### 3.1 Implementation Logic
The method `extract_best_snippet_neural` in `searcher.py` performs the following steps:

1.  **Segmentation:** The full document text is split into logical paragraph units using a custom delimiter strategy (`_split_into_paragraphs`).
2.  **Cross-Encoder Reuse:** We efficiently reuse the loaded **Cross-Encoder** model. Instead of scoring the whole document, we construct pairs of `[User Query, Paragraph]` for the text segments.
3.  **Selection:** The model predicts a relevance score for each paragraph.
4.  **Extraction:** The system returns the single paragraph with the highest semantic similarity score.

### 3.2 Impact on User Experience
This feature ensures that the user sees the exact passage that answers their question directly on the results list, rather than a generic abstract. This significantly reduces the cognitive load required to evaluate search results.

---

# 4. Further AI Enhancement B: AI-Powered Metadata (Auto-Tagging)

As a second AI enhancement, we leveraged the **Zero-Shot capabilities** of the LLM (Gemini 2.5 Flash) to generate structured metadata from unstructured text on the fly.

### 4.1 Implementation Logic
The `generate_document_tags` method analyzes the document content to produce categorization labels.

1.  **Optimization & Relevance Strategy:** We deliberately truncate the input to the **first 2,000 characters**.
    *   **Token Efficiency:** This drastically reduces token consumption, preventing the exhaustion of API quotas on long-tail content.
    *   **Structural Sufficiency:** Encyclopedic articles (like Wikipedia) follow an "inverted pyramid" structure, where the introductory section defines the subject matter. Therefore, the beginning of the document provides sufficient context for the model to extract accurate tags and categories without needing to process the full text.

2.  **Structured Prompting:** We use a strict prompt to force the LLM into a classification role, requiring it to output:
    *   **Main Category:** A high-level classification (e.g., "History", "Science", "Biography").
    *   **Keywords:** A list of 5 specific tags relevant to the content.

3.  **Parsing:** The system programmatically parses the LLM's text output into a JSON object (`{"category": "...", "tags": [...]}`).

### 4.2 UI Integration & Utility
This feature is integrated directly into the **Document Details view** of our web interface. When a user clicks to view a specific document, the system calls the LLM in real-time to generate these labels. This provides the user with an immediate high-level understanding of the document's topics without needing to read the entire text, facilitating faster information filtering.

---

# Conclusion

This project successfully achieved the primary objective of Assignment 2: upgrading a classical Information Retrieval system into a modern **Semantic Search Engine**. By transitioning from a purely lexical approach (BM25) to a hybrid neural architecture, the system can now bridge the "vocabulary gap"—understanding user intent even when query terms do not explicitly match document tokens.

Key achievements and architectural insights include:

1.  **The Power of Two-Stage Retrieval:**
    We demonstrated that replacing BM25 entirely is not necessary to achieve high precision. Instead, using BM25 as an efficient first-stage filter, followed by a **Neural Reranker (`Cross-Encoder`)**, provides the best balance between speed and accuracy. The implementation of the **`unicamp-dl/mMiniLM-L6-v2-pt-v2`** model allowed us to capture semantic nuances in Portuguese (such as synonyms and polysemy) that the previous implementation missed, significantly improving the quality of the Top-10 results.

2.  **RAG as a Paradigm Shift:**
    The integration of **Retrieval-Augmented Generation (RAG)** fundamentally changed the system's utility. By injecting the **Top-5 reranked documents** into the **Gemini 2.5 Flash** context, we transformed the user experience from simply "finding documents" to "getting answers." The engineering decision to use the Flash model proved critical: it allowed us to maintain the large context window (1M tokens) required for multi-document reasoning while ensuring the high throughput (RPM) necessary for a responsive search interface.

3.  **AI Beyond Ranking:**
    Through the "Further AI Enhancements," we showed that LLMs and Cross-Encoders can improve the User Interface (UI) itself.
    *   **Semantic Snippets** solved the issue of generic document descriptions by dynamically extracting the most relevant paragraph.
    *   **Auto-Tagging** demonstrated the power of Zero-Shot classification, organizing unstructured text into structured categories without manual intervention.

In summary, the system has evolved from a simple term-matching script into a robust, AI-powered retrieval pipeline. The chosen models—optimized for efficiency rather than raw size—demonstrate a realistic engineering approach to building scalable Search & QA systems.
