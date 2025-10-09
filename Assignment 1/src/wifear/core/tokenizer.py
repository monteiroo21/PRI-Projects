class Tokenizer:
    def __init__(self):
        self.tokens = []


    def tokenize(self, text):
        self.tokens = text.split()
        return self.tokens