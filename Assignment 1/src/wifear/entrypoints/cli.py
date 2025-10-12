import argparse

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

    indexer.index_documents(args.file_path)
    indexer.merge_blocks()


if __name__ == "__main__":
    main()
