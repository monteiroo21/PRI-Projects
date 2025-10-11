import re
from typing import List
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk

# Ensure stopwords list is downloaded
nltk.download("stopwords", quiet=True)


class PortugueseTokenizer:
    """
    Simple tokenizer for the Portuguese language.
    - Converts text to lowercase
    - Removes punctuation and numbers
    - Removes stopwords
    - Applies stemming (SnowballStemmer)
    - Filters tokens by minimum length
    """

    def __init__(self, min_len: int = 3):
        self.stemmer = SnowballStemmer("portuguese")
        self.stopwords = set(stopwords.words("portuguese"))
        self.min_len = min_len
        # Regex pattern for alphabetic words including accented characters
        self.token_pattern = re.compile(r"\b[a-záàâãéèêíïóôõöúçñ]+\b", re.IGNORECASE)

    def tokenize(self, text: str) -> List[str]:
        # Convert text to lowercase
        text = text.lower()

        # Extract alphabetic tokens (keep accented characters)
        tokens = self.token_pattern.findall(text)

        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stopwords and len(t) >= self.min_len]

        # Apply stemming
        stems = [self.stemmer.stem(t) for t in tokens]

        return stems


# # Quick test example
# if __name__ == "__main__":
#     tokenizer = PortugueseTokenizer(min_len=3)
#     text = "Os carros elétricos estão a tornar-se populares em Portugal."
#     print(tokenizer.tokenize(text))
