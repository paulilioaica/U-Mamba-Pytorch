import torch
from torch.utils.data import Dataset 
import urllib.request
import os

class ShakespeareDataset(Dataset):  
    def __init__(self, text, char_to_index, sequence_len=30): 
        self.char_to_index = char_to_index 
        sequences = []  
        for i in range(0, len(text)-sequence_len, sequence_len):  
            sequences.append(text[i:i+sequence_len+1])  
        self.sequences = sequences  
  
    def __len__(self):  
        return len(self.sequences)  
  
    def __getitem__(self, index):  
        sequence = self.sequences[index]  
        x = torch.tensor([self.char_to_index[c] for c in sequence[:-1]], dtype=torch.long)  
        y = torch.tensor([self.char_to_index[c] for c in sequence[1:]], dtype=torch.long)  
        return x, y  
    

def get_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    filename = "shakespeare.txt"
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

    # Load the text data and tokenize it
    with open(filename) as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}
    return text, chars, char_to_index, index_to_char