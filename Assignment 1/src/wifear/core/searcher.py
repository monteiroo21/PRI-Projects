import json
import math

from wifear.core.tokenizer import PortugueseTokenizer


class SearchEngine:
    def __init__(self, index_path, tokenizer: PortugueseTokenizer):
        # Load the index and set the tokenizer upon initialization
        self.index = self.load_index(index_path)
        self.tokenizer = tokenizer
        self.avg_doc_len = self.calculate_avg_doc_len()

    def load_index(self, index_path):
        """Load the pre-built inverted index from disk."""
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        print("Index loaded successfully.")
        return index

    def calculate_avg_doc_len(self):
        """Calculate the average document length for the corpus."""
        total_len = 0
        num_docs = 0

        for term, postings in self.index.items():
            for doc_id, positions in postings.items():
                total_len += len(positions)
                num_docs += 1

        avg_len = total_len / num_docs if num_docs else 0
        print(f"Average document length: {avg_len}")
        return avg_len

    def idf(self, term):
        """Calculate the IDF (Inverse Document Frequency) for a term."""
        df = len(self.index.get(term, {}))  # Document frequency of the term
        N = len(self.index)  # Total number of documents in the index
        if df == 0:
            return 0
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def bm25_score(self, query_terms, doc_id, k1=1.2, b=0.75):
        """Calculate the BM25 score for a document given the query."""
        score = 0
        for term in query_terms:
            if term in self.index:
                f = self.index[term].get(doc_id, 0)  # Term frequency in document
                doc_len = len(self.index[term].get(doc_id, []))
                idf = self.idf(term)
                score += (
                    idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
                )
        return score

    def query(self, query_text):
        """Search for documents that match the given query text."""
        # Tokenize the query
        query_terms = self.tokenizer.tokenize(query_text)

        result_docs = {}

        # Process each query term and calculate BM25 scores
        for doc_id in self.index:
            score = self.bm25_score(query_terms, doc_id)
            if score > 0:
                result_docs[doc_id] = score

        # Sort results by BM25 score in descending order
        sorted_results = sorted(result_docs.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


# Usage example:
# Initialize the tokenizer
tokenizer = PortugueseTokenizer(min_len=3)

# Initialize the SearchEngine with the path to your index and the tokenizer
search_engine = SearchEngine(index_path="data/index_final.json", tokenizer=tokenizer)

# Example query
query_text = "freguesia Amares"  # Replace with the user's query
query_results = search_engine.query(query_text)

print("Query Results:", query_results)
