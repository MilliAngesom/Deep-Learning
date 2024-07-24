import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv
from collections import Counter
import numpy as np
import json


# # to save and load the vocabulary stoi and itos for later use
# def save_vocab_to_json(vocab, filename):
#     with open(filename, 'w', encoding='utf-8') as file:
#         json.dump({'stoi': vocab.stoi, 'itos': vocab.itos}, file, ensure_ascii=False, indent=4)





# Instance class
class Instance:
    def __init__(self, text, label):
        self.text = text
        self.label = label


class NLPDataset(Dataset):
    def __init__(self, filename,text_vocab={}, label_vocab={}):
        self.text_vocab= text_vocab
        self.label_vocab =label_vocab
        self.instances = []
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                # Because the format is 'text, label'
                self.instances.append(Instance(row[0], row[1]))

    def __len__(self):
        return len(self.instances)

    # def __getitem__(self, idx):
    #     return self.instances[idx]
    def __getitem__(self, idx):
        instance = self.instances[idx]
        numericalized_text = self.text_vocab.encode(instance.text.split())
        numericalized_label = self.label_vocab.encode(instance.label.split())
        return torch.tensor(numericalized_text, dtype=torch.long), torch.tensor(numericalized_label, dtype=torch.long)
    

# Function to build a frequency dictionary from the dataset
def build_frequency_dict(dataset):
    text_freqs = Counter()
    label_freqs = Counter()
    for instance in dataset.instances:
        text_freqs.update(instance.text.split())
        label_freqs.update(instance.label.split())
    return text_freqs, label_freqs


#Vocab class
class Vocab:
    def __init__(self, freqs, indicator, max_size=-1, min_freq=1 ):
        self.indicator = indicator
        if indicator == 'text':
            self.itos = {0: "<PAD>", 1: "<UNK>"}
            self.stoi = {"<PAD>": 0, "<UNK>": 1}
        elif indicator == 'label':
            self.itos = {}
            self.stoi = {}
        # Create a frequency dictionary from the tokens
        # freqs = Counter(token_freqs)
        freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        
        # Apply max size and min frequency constraints
        for word, freq in freqs:
            if freq < min_freq and max_size > 0 or len(self.itos) >= max_size and max_size > 0:
                break
            self.stoi[word] = len(self.itos)
            self.itos[self.stoi[word]] = word
    # Given word or sentence returns the index of the word/s
    def encode(self, sequence):
        if isinstance(sequence, str):
            if self.indicator == 'text':
                return self.stoi.get(sequence, self.stoi["<UNK>"])
            else:
                return self.stoi.get(sequence)
        else:
            if self.indicator == 'text':
                return [self.stoi.get(word, self.stoi["<UNK>"]) for word in sequence]
            else:
                return [self.stoi.get(word) for word in sequence]
    # Given index returns the word
    def decode(self, indices):
        if self.indicator == 'text':
            return [self.itos.get(index, "<UNK>") for index in indices]
        else:
            return [self.itos.get(index) for index in indices]




# def load_vocab_from_json(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     vocab = Vocab({}, '')  # Create a dummy Vocab object
#     vocab.stoi = data['stoi']
#     vocab.itos = data['itos']
#     return vocab


# text_vocab = load_vocab_from_json('text_vocab.json')
# label_vocab = load_vocab_from_json('label_vocab.json')




# save_vocab_to_json(text_vocab, 'million_text_vocab.json')
# save_vocab_to_json(label_vocab, 'million_label_vocab.json')



def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def create_embedding_matrix(vocab, glove_embeddings, embedding_dim=300):
    matrix_len = len(vocab.stoi)
    weights_matrix = np.zeros((matrix_len, embedding_dim))

    for word, i in vocab.stoi.items():
        try: 
            weights_matrix[i] = glove_embeddings[word]
        except KeyError:
            np.random.seed(7052020) 
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

    # Ensure padding token is zero
    weights_matrix[0] = np.zeros((embedding_dim, ))
    return torch.tensor(weights_matrix, dtype=torch.float32)


train_dataset = NLPDataset('sst_train_raw.csv')#,text_vocab,label_vocab)
text_freqs, label_freqs = build_frequency_dict(train_dataset)

# Use token_freqs with the Vocab class
text_vocab = Vocab(text_freqs, 'text', max_size=-1, min_freq=1)
label_vocab = Vocab(label_freqs, 'label', max_size=-1, min_freq=1)
train_dataset = NLPDataset('sst_train_raw.csv',text_vocab,label_vocab)


glove_embeddings = load_glove_embeddings('sst_glove_6b_300d.txt')
embedding_matrix = create_embedding_matrix(text_vocab, glove_embeddings)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=True)








def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    
    # Padding the sequences with pad_index
    padded_texts = pad_sequence([torch.tensor(text) if not isinstance(text, torch.Tensor) else text for text in texts], 
                            batch_first=True, padding_value=pad_index)

    
    labels = torch.tensor(labels)
    return padded_texts, labels, lengths




batch_size = 10
shuffle = True
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi["<PAD>"]))
texts, labels, lengths = next(iter(train_dataloader))
# print(f"Texts: {texts}")
# print(f"Labels: {labels}")
# print(f"Lengths: {lengths}")
