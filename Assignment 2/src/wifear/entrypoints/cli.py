import argparse
import time

from wifear.core.limit_memory import start_memory_monitor
from wifear.core.logger import setup_logging
from wifear.core.spimi import SPIMIIndexer
from wifear.core.tokenizer import PortugueseTokenizer

setup_logging()
start_memory_monitor(show_memory_updates=True)


def main():
    parser = argparse.ArgumentParser(description="Wifear Indexer CLI")
    parser.add_argument("file_path", type=str, help="Path to the file to index")

    args = parser.parse_args()

    tokenizer = PortugueseTokenizer()
    indexer = SPIMIIndexer(tokenizer)

    tic1 = time.time()
    indexer.index_documents(args.file_path)
    tic = time.time()
    indexer.merge_blocks()
    toc = time.time()
    print(f"[SPIMI] Indexing completed in {tic - tic1:.2f} seconds.")
    print(f"[SPIMI] Merging completed in {toc - tic:.2f} seconds.")
    print(f"[SPIMI] Total processing time: {toc - tic1:.2f} seconds.")


if __name__ == "__main__":
    main()
