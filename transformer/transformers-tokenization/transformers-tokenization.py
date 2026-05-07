import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        lis = []
        for text in texts :
            for word in text.lower().split():
                lis.append(word)
        lis = list(set(lis))
        lis.sort()
        for i in range(len(lis)):
            if lis[i] not in self.word_to_id:
                self.word_to_id[lis[i]] = i + 4
        self.word_to_id[self.pad_token] = 0 
        self.word_to_id[self.unk_token] = 1 
        self.word_to_id[self.bos_token] = 2 
        self.word_to_id[self.eos_token] = 3 
        self.id_to_word = {self.word_to_id[key] : key for key in self.word_to_id.keys()}
        self.vocab_size = len(lis) + 4
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if len(text) == 0 :
            return []
        
        words = text.lower().split(' ')
        encoded = []
        # encoded.append(self.word_to_id)
        for word in words:
            encoded.append(self.word_to_id.get(word , 1))
        return encoded
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = ""
        for tok in ids :
            decoded += self.id_to_word.get(tok, self.unk_token) + " "
        return decoded[: -1]
        # YOUR CODE HERE
