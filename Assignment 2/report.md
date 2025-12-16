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