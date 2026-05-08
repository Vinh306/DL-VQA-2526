from underthesea import word_tokenize as vi_word_tokenize
import re
from collections import Counter

class TextProcessor:
    @staticmethod
    def tokenize(text):
        return vi_word_tokenize(text)
    
    @staticmethod
    def preprocess(text):
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
class Vocabulary:
    class Vocabulary:
        def __init__(self, freq_threshold=2):
            self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.stoi = {v: k for k, v in self.itos.items()}
            self.freq_threshold = freq_threshold

        def __len__(self):
            return len(self.itos)

        def build_vocab(self, sentence_list):
            freq = Counter()
            idx = 4
            for sent in sentence_list:
                for w in TextProcessor.tokenize(sent):
                    freq[w] += 1
                    if freq[w] == self.freq_threshold:
                        self.stoi[w] = idx
                        self.itos[idx] = w
                        idx += 1

        def numericalize(self, text):
            return [self.stoi.get(w, self.stoi["<UNK>"]) for w in TextProcessor.tokenize(text)]