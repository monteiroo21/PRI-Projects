## Project Description

This project implements a complete **Information Retrieval (IR) system** capable of indexing and searching a large-scale document collection — specifically, the **Portuguese Wikipedia dump** (`ptwiki-articles-with-redirects.jsonl`).

The system follows a modular architecture comprising three main components:

1. **Reader and Preprocessor (`reader.py`)**
   This module cleans and filters the raw corpus, removing redirects and empty documents. It produces two JSONL files:

   - `ptwiki_clean.jsonl` — containing the cleaned articles ready for indexing.
   - `docstore.jsonl` — a lightweight metadata file storing document IDs, titles, and short descriptions for fast lookup during search.

2. **Indexer (`spimi.py`)**
   Implements the **Single-Pass In-Memory Indexing (SPIMI)** algorithm to build a **positional inverted index** under a 2 GB memory constraint.

   - The indexer tokenizes documents using a **custom Portuguese tokenizer** (`tokenizer.py`), which applies lowercasing, stopword removal, stemming (via `nltk.SnowballStemmer`), and token length filtering.
   - Documents are processed in parallel chunks to produce compressed partial indexes (`block_XXX.json.gz`), which are later merged into a single consolidated index (`index_final.jsonl`) using a minimum document frequency threshold.

3. **Searcher (`searcher.py`)**
   Loads the final index into an **SQLite database** (`index.db`) and performs efficient **BM25 ranking** of documents based on query terms.

   - The searcher supports configurable BM25 parameters (`k1`, `b`), precomputes IDF values, and builds a **forward index** (term-frequency map per document) for relevance feedback.
   - The relevance feedback mechanism allows retrieving documents **similar to a given one**, emulating “search by example” functionality.

---

## How to Run the Project

To execute the full information retrieval pipeline — from corpus preprocessing to the running search API — follow the steps below:

1. **Clean and preprocess the dataset**

   ```bash
   uv run python -m wifear.reader
   ```

   This command filters the raw Wikipedia dump, removes redirects and empty pages, and generates two files:

   - `data/ptwiki_clean.jsonl` — cleaned corpus for indexing.
   - `data/docstore.jsonl` — metadata file for document titles and descriptions.

2. **Build the inverted index**

   ```bash
   uv run python -m wifear.entrypoints.cli data/ptwiki_clean.jsonl
   ```

   Runs the SPIMI indexer, which tokenizes documents, builds partial indexes, and merges them into a final positional inverted index (`data/index_final.jsonl`).

3. **Load the index into SQLite**

   ```bash
   uv run python -m wifear.core.load_db
   ```

   Imports the consolidated inverted index into a local SQLite database (`index.db`) for fast query access by the search engine.

4. **Start the web search interface**

   ```bash
   uv run uvicorn wifear.entrypoints.asgi:app --reload
   ```

   Launches the FastAPI application exposing REST endpoints for querying and relevance feedback.
   The interface can be accessed locally at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## Reader

The **Reader** module (`reader.py`) is responsible for the **initial corpus preprocessing** stage.
It reads the raw Portuguese Wikipedia dump (`ptwiki-articles-with-redirects.jsonl`), filters irrelevant records, and outputs a clean and structured dataset ready for indexing.

### Main Responsibilities

1. **Filtering and Cleaning**

   - Removes redirect pages (`redirect = true`) and documents with empty or invalid text fields.
   - Ensures that all `out_links` are stored as valid Python lists.
   - Ignores malformed or corrupted JSON lines gracefully.

2. **Batch Processing for Large Datasets**

   - Processes the dataset **line by line** using buffered batches of 50,000 records to avoid excessive memory consumption.
   - Each batch is written incrementally to disk, ensuring scalability for multi-gigabyte corpora.

3. **Outputs**

   - **`ptwiki_clean.jsonl`** — contains the cleaned and filtered articles to be used by the SPIMI indexer.
   - **`docstore.jsonl`** — stores document metadata (ID, title, description) for quick lookup and display during search results.

4. **Execution Flow**
   The entry point of the module executes the full preprocessing pipeline:

   ```python
   if __name__ == "__main__":
       read_jsonl_to_jsonl_in_batches(INPUT_JSONL, OUTPUT_JSONL, batch_size=BATCH_SIZE, limit=LIMIT)
   ```

   This function reads the source JSONL file, applies the cleaning rules, and writes the resulting cleaned dataset to the output directory.

### Example Console Output

When executed, the script displays progress logs such as:

```
Processed 66,625 lines → total written: 50,000
Processed 140,987 lines → total written: 100,000
...
Saved 1,154,228 cleaned documents to: $USER_PATH/data/ptwiki_clean.jsonl
```

---

## Tokenizer

The **Tokenizer** (`tokenizer.py`) is a lightweight yet effective text processing module specifically designed for the **Portuguese language**.
It performs lexical normalization and prepares text for indexing by applying a series of linguistic transformations that improve retrieval consistency and reduce vocabulary size.

### Main Responsibilities

1. **Text Normalization**

   - Converts all text to lowercase to ensure case-insensitive matching.
   - Extracts only **alphabetic tokens**, including accented Portuguese characters, using a custom regular expression (`[a-záàâãéèêíïóôõöúçñ]+`).
   - Removes numbers, punctuation, and other non-alphabetic symbols.

2. **Stopword Removal**

   - Utilizes the built-in list of **Portuguese stopwords** from the `nltk.corpus.stopwords` module.
   - Eliminates common functional words (e.g., _“de”, “para”, “em”, “os”_) that do not contribute to semantic meaning or ranking.

3. **Stemming**

   - Applies **Snowball stemming** for Portuguese using `nltk.SnowballStemmer("portuguese")`.
   - Reduces words to their root form (e.g., _“cidades”, “cidade” → “cidad”_), allowing the index to treat morphological variants as the same term.

4. **Token Filtering**

   - Discards tokens shorter than a configurable minimum length (default: **3 characters**).
   - This reduces noise and helps focus on meaningful content-bearing words.

### Example Workflow

```python
from wifear.core.tokenizer import PortugueseTokenizer

tokenizer = PortugueseTokenizer(min_len=3)
tokens = tokenizer.tokenize("A cidade de Lisboa é a capital de Portugal.")
print(tokens)
```

**Output:**

```
['cidad', 'lisbo', 'capit', 'portugal']
```

### Design Considerations

- The tokenizer is designed to be **consistent** across all system components — it is used both by the **SPIMI indexer** and the **Searcher**, ensuring identical preprocessing for documents and user queries.

---

## SPIMI

The **SPIMI Indexer** (`spimi.py`) implements the **Single-Pass In-Memory Indexing** algorithm, designed to construct an efficient **positional inverted index** over a large-scale corpus while respecting a **2 GB memory limit**.
Instead of holding the entire dataset in memory, the SPIMI approach builds sorted **partial index blocks** that are later merged into a single global index (`index_final.jsonl`).

This implementation was carefully optimized for **parallel processing**, **low memory usage**, and **fast disk I/O**, ensuring that the indexing pipeline can handle the Portuguese Wikipedia corpus (~2.7M documents).

---

### Indexer

#### Overview

The `index_documents()` method is the **main coordination function** of the SPIMI pipeline.
It manages the entire lifecycle of the indexing process — reading the corpus, distributing work to parallel processes, monitoring memory, and saving compressed partial blocks to disk.

Each worker process runs the helper function `process_chunk()`, which handles tokenization and block creation independently.

---

### 1. Purpose of `index_documents`

The primary goal of the `index_documents()` function is to incrementally process the document collection in small batches and generate multiple **compressed partial inverted indexes** (`block_XXX.json.gz`), ensuring that the system never exceeds the defined memory threshold.

It performs the following tasks:

1. Reads the cleaned corpus file (`ptwiki_clean.jsonl`) line by line.
2. Groups documents into manageable **chunks** (default: 5,000 documents per batch).
3. Launches multiple worker processes via Python’s `multiprocessing.Pool`.
4. Passes the tokenizer configuration to each worker.
5. Writes each partial block to disk immediately after creation.
6. Records key metadata, such as the number of documents processed, average document length, and total number of blocks created.

---

### 2. Streaming the Document Collection

The implementation follows a **streaming-based design** — only a small subset of documents is held in memory at any time.
Each line of the corpus is parsed into a JSON object, validated, and accumulated into a temporary buffer until the chunk limit is reached.

This approach provides several advantages:

- Prevents out-of-memory errors, even with millions of documents.
- Ensures predictable memory usage determined by `chunk_size`.
- Enables early block flushing and continuous progress tracking.

Example log during indexing:

```
[SPIMI] Processed 25,000 docs so far...
[Worker 003] Block written (18,524 terms) → index_blocks/block_003.json.gz
```

---

### 3. Parallel Chunk-Based Indexing

Each chunk of documents is processed in **parallel** by multiple workers (`max_parallel = min(cpu_count(), 3)`).

Each worker:

- Tokenizes its assigned subset using the **PortugueseTokenizer**.
- Builds a **positional inverted index**, recording every term’s position within each document.
- Sorts the terms alphabetically for efficient merging later.
- Saves its block in compressed JSON (`gzip`) format.

Example structure of a block:

```json
{
  "portugal": {"12": [3, 9, 14], "45": [7]},
  "cidad": {"18": [5, 13], "21": [2, 8, 16]},
  ...
}
```

Each block represents a self-contained index, fully sorted and ready to be merged.
Because each worker operates independently, the indexing process fully utilizes available CPU cores without synchronization overhead.

---

### 4. Memory Monitoring

Memory consumption is monitored continuously using the **psutil** library.
The private method `_memory_full()` checks whether current memory usage has reached 90% of the assigned limit (2 GB by default).
If the threshold is exceeded:

- The current chunk is flushed immediately.
- A new block is started.
- The program exits gracefully if the global limit is surpassed.

This ensures deterministic, memory-safe behavior throughout indexing.

---

### 5. Metadata Generation

At the end of the indexing phase, the indexer writes a small `metadata.json` file containing global statistics:

```json
{
  "num_docs": 1154228,
  "avg_doc_len": 218.46,
  "num_blocks": 245
}
```

This metadata is later used by the **Searcher** to precompute BM25 parameters such as average document length.

---

## Merge

The **merging phase** is handled by the `merge_blocks()` function.
It consolidates all partial SPIMI blocks into a single **global positional index** (`data/index_final.jsonl`), applying frequency-based term filtering and maintaining sorted order.

---

### 1. Multiway Merge Process

The `merge_blocks()` function implements a **multiway merge algorithm**, similar to the merge step in mergesort.
It works as follows:

1. Opens all block files (`block_XXX.json.gz`) simultaneously as input streams.
2. Iterates through their term–posting pairs in lexicographic order using `heapq.merge`.
3. When the same term appears across multiple blocks:

   - Their postings are concatenated.
   - Document IDs are merged and sorted.

4. Writes the merged postings incrementally to the final output file in JSONL format.

This design ensures that:

- Only one term per block resides in memory at a time.
- The merge operation scales linearly with the number of terms.
- Memory footprint remains small and predictable.

Example of final output line:

```json
{ "portugal": { "12": [3, 9, 14], "45": [7], "58": [2, 17] } }
```

---

### 2. Minimum Term Frequency Filtering

During merging, a **minimum document frequency threshold** (`min_df`) is applied (default: 3).
This filter removes rare terms and reduces index size without affecting retrieval quality.

Steps:

1. Compute the number of documents in which each term appears.
2. Discard terms with `df < min_df`.
3. Write only the remaining terms to the final index file.

Benefits:

- Removes noisy or misspelled words.
- Reduces index size and disk usage.
- Improves query speed and precision.

---

### 3. Output Format and Efficiency

The final index (`index_final.jsonl`) follows a **line-oriented JSONL structure** for high-throughput sequential access:

- Each line represents one term and its complete posting list.
- The format supports direct streaming into SQLite during the `load_db.py` phase.
- The writing process is buffered to minimize disk I/O overhead.

Example output snippet:

```
{"am": {"2": [4, 19], "8": [12]}}
{"fregues": {"3": [6, 7], "5": [10, 21, 45]}}
{"portugal": {"1": [3, 9, 14], "7": [6, 22]}}
```

---

## Searcher

The **Searcher** (`searcher.py`) is the core component responsible for transforming the indexed data into **useful, ranked search results**.
It represents the **retrieval and ranking stage** of the information retrieval pipeline, where user queries are matched against the precomputed inverted index to identify and score the most relevant documents.

Unlike the indexer — which builds the inverted index from scratch — the Searcher **loads and operates entirely in memory**, enabling **real-time query responses**. It also supports **advanced retrieval features** such as **relevance feedback** and similarity-based document comparison.

---

### Main Responsibilities

1. **Index Loading**

   - The Searcher connects to the local **SQLite database** (`index.db`) generated by the `load_db.py` script.
   - It reads the `inverted_index` table, which contains a JSON representation of each term’s posting list — mapping terms to document IDs and positional occurrences within the text.
   - Each posting list is deserialized from JSON and loaded into a Python dictionary of the form:

     ```python
     {
         "fregues": {12: [2, 18, 56], 45: [14, 29]},
         "am": {1: [5, 6], 3: [44]},
         ...
     }
     ```

   - During loading, document identifiers are converted to integers and malformed entries are skipped to ensure robustness.
   - The entire index is then kept **in-memory**, allowing extremely fast query processing and term access.

2. **Precomputation of Global Statistics**

   - After loading, the Searcher precomputes several key statistics used by the **BM25** ranking model:

     - **Document lengths (DL):** total number of term occurrences per document.
     - **Average document length (avgDL):** used for BM25 length normalization.
     - **Inverse Document Frequency (IDF):** measures term rarity across documents.

   - These statistics are cached for all terms, reducing computational overhead during query execution.

3. **BM25 Ranking**

   - The system implements the **BM25 algorithm**, a probabilistic ranking function widely adopted in modern search engines.
   - For each term in the query, BM25 computes a weighted score reflecting its contribution to document relevance, combining:

     - **Term frequency (TF):** how often the term appears in the document.
     - **Document length normalization:** longer documents are penalized.
     - **Inverse document frequency (IDF):** rarer terms get higher weight.

   - The BM25 score for a document `d` and query term `t` is computed as:
     [
     \text{score}(t, d) = \text{IDF}(t) \cdot \frac{TF(t, d) \cdot (k_1 + 1)}{TF(t, d) + k_1 \cdot (1 - b + b \cdot \frac{DL(d)}{avgDL})}
     ]
     where (k_1 = 1.2) and (b = 0.75) are the default hyperparameters.

4. **Query Processing**

   - When a user submits a query, it is first passed through the **PortugueseTokenizer** to ensure consistent preprocessing (lowercasing, stemming, and stopword removal).
   - For each tokenized query term, the Searcher retrieves its posting list, calculates BM25 scores across all relevant documents, and aggregates them into a final score dictionary.
   - Results are sorted by descending BM25 score and returned along with their metadata (title, snippet, and score).

5. **Relevance Feedback and Similarity Search**

   - The Searcher also supports **pseudo-relevance feedback** through the `like_document()` method.
   - This method allows users to find documents **similar to a specific one** — mimicking the “Find similar pages” or “Search by example” features seen in commercial search engines.
   - Internally, it analyzes the term-weight distribution of a document, expands it into a pseudo-query using its most representative terms, and retrieves other documents sharing similar patterns.

6. **Docstore Integration**

   - Alongside the inverted index, the Searcher loads a lightweight **Docstore** (`data/docstore.jsonl`) containing document metadata such as title and a short description (first text lines).
   - This allows the system to **return human-readable results** directly from the backend, without requiring an additional lookup layer.

7. **Performance**

   - Once initialized, the Searcher operates entirely **in-memory**, leveraging **NumPy arrays** for efficient vectorized ranking and sorting operations.
   - This results in sub-second response times, even when dealing with hundreds of thousands of documents.

---

### Load DB

The **load_db.py** module is the bridge between the indexing and retrieval stages.
It converts the JSONL-based inverted index (`data/index_final.jsonl`) into an **SQLite** database (`index.db`), allowing structured and fast loading by the Searcher.

#### Execution Steps

1. Creates a table named `inverted_index` with the following schema:

   ```sql
   CREATE TABLE IF NOT EXISTS inverted_index (
       term TEXT PRIMARY KEY,
       postings TEXT
   );
   ```

2. Iterates through each JSONL entry and inserts `(term, postings)` pairs into the table in batches of 1,000 entries.
3. Commits transactions periodically to minimize memory usage.
4. On completion, it prints a summary message:

   ```
   Done importing index into SQLite
   ```

This conversion makes the retrieval phase **modular and portable**, since the SQLite file can be distributed and queried independently.

---

### NumPy

The use of **NumPy** is critical for achieving real-time performance during ranking.
When the BM25 scores are computed, the Searcher converts Python dictionaries into NumPy arrays to perform efficient vectorized operations.

Example workflow:

```python
doc_ids = np.fromiter(scores.keys(), dtype=int)
doc_scores = np.fromiter(scores.values(), dtype=float)
top_idx = np.argsort(-doc_scores)[:top_k]
```

This approach eliminates Python’s iteration overhead, enabling:

- **Fast top-k selection** (e.g., top 10 results per query).
- Efficient numeric computation for large candidate sets.
- Improved scalability with minimal additional memory footprint.

By integrating NumPy, the Searcher achieves a balance between **algorithmic accuracy** and **computational speed**, crucial for real-world search systems.

---

### Forward Index

In addition to the inverted index, the Searcher automatically builds a **forward index** — a structure mapping each document ID to its terms and corresponding frequencies.

Example structure:

```python
{
    1: {"portugal": 3, "lisbo": 5, "cidad": 2},
    42: {"am": 4, "habit": 3, "fregues": 1},
    ...
}
```

#### Purpose

- Enables **relevance feedback** and **similarity search**, since the Searcher can directly analyze a document’s term distribution.
- Facilitates the computation of **term importance weights** without repeatedly scanning the inverted index.
- Allows the system to identify the most influential terms in a document — the ones that best represent its content.

#### Internal Integration

The forward index is generated dynamically when the Searcher loads the SQLite data:

```python
for term, postings in self.index.items():
    for doc_id, positions in postings.items():
        self.forward_index[doc_id][term] = len(positions)
```

This ensures consistency between both index types and guarantees that all derived statistics remain synchronized.

---

## Conclusion

The development of this project resulted in a **complete Information Retrieval (IR) system** capable of processing, indexing, and searching a large-scale document collection — specifically, the **Portuguese Wikipedia corpus**.
The work covered all the essential stages of a modern search engine, from **data acquisition and preprocessing**, through **efficient indexing**, to **fast and accurate information retrieval**.

The system follows a **modular architecture**, composed of four main components — **Reader**, **Tokenizer**, **SPIMI Indexer**, and **Searcher** — that operate in a fully integrated and coherent way.
The **Reader** ensures data quality by cleaning and filtering the corpus, removing irrelevant or malformed entries.
The **Tokenizer**, designed for Portuguese, standardizes lexical forms by applying lowercasing, stopword removal, and stemming.
The **SPIMI Indexer** enables incremental construction of the inverted index within a 2 GB memory limit, using streaming and parallel processing for scalability.
Finally, the **Searcher** implements the **BM25 ranking model**, providing efficient and accurate retrieval as well as similarity-based search through relevance feedback mechanisms.

This modular design achieves a strong balance between **efficiency, scalability, and precision**, allowing the system to handle millions of documents while maintaining low query latency.
Through techniques such as memory monitoring, streaming computation, and NumPy-based vectorized ranking, the system delivers reliable performance consistent with real-world IR requirements.

Beyond its technical aspects, this project offered valuable insight into the **core principles of Information Retrieval**, including the construction and optimization of inverted indexes, the linguistic normalization of text, and statistical relevance models.
By translating these theoretical foundations into an operational system, the project reinforced both algorithmic understanding and practical data engineering skills.

In summary, the project achieved its main objectives by producing a **functional, extensible, and high-performance search engine**, aligned with the theoretical and practical goals of the _Information Processing and Retrieval (PRI)_ course.
It establishes a solid foundation for future extensions, such as **semantic search, neural embeddings, or machine learning–based ranking**, paving the way for more intelligent and context-aware retrieval systems.
