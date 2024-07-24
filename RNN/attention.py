import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loading import*
from baseline_model import*

torch.manual_seed(705200)

# Attention module as per Bahdanau et al.
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.W2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.V = nn.Linear(hidden_dim // 2, 1)

    def forward(self, encoder_outputs, lengths):
        # Apply linear transformation and nonlinearity (tanh) to encoder outputs
        transformed_outputs = torch.tanh(self.W1(encoder_outputs))

        # Calculate scores
        scores = self.V(transformed_outputs)
        scores = scores.squeeze(2)

        # Create mask based on lengths to ignore padding
        mask = torch.arange(encoder_outputs.size(1))[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float('-inf'))

        # Apply softmax to scores to get attention weights
        attn_weights = F.softmax(scores, dim=1)

        # Multiply weights by encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        return context_vector, attn_weights

# RNN model with optional Bahdanau attention
class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers, rnn_cell_type='gru', 
                 bidirectional=False, dropout_rate=0.5, use_attention=False):
        super(RNNModel, self).__init__()
        self.embedding = embedding_layer
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define the RNN layer
        rnn_cell = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}.get(rnn_cell_type.lower())
        self.rnn = rnn_cell(input_size=embedding_layer.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True,
                            bidirectional=bidirectional)

        # Define the attention layer if use_attention is True
        if self.use_attention:
            self.attention = BahdanauAttention(hidden_dim * (2 if bidirectional else 1))

        # Define two fully connected layers
        self.fc1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # Embedding layer
        x = self.embedding(x)

        # RNN layer
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(x_packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention layer
        if self.use_attention:
            output, attn_weights = self.attention(output, lengths)
        else:
            # Without attention, just take the outputs from the last time step
            # output = output[torch.arange(output.size(0)), lengths - 1]
            tensor1 = output[torch.arange(output.size(0)), lengths - 1, :self.hidden_dim]
            tensor2 = output[torch.arange(output.size(0)), 0, self.hidden_dim:]
            output = torch.cat((tensor1, tensor2), dim=1)

        # Fully connected layers
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output











# training loop
def train(model, data_loader, optimizer, criterion, clip_value):
    model.train()
    for texts, labels, lengths in data_loader:
        optimizer.zero_grad()
        labels = labels.view(-1, 1)
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels.float())
        loss.backward()
        # Clip gradients to avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()



def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            labels = labels.view(-1, 1)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            preds = torch.sigmoid(outputs).round()
            all_labels.extend(labels.view(-1).tolist())
            all_preds.extend(preds.view(-1).tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, f1, conf_matrix



# Main function
def main():
    # Load train dataset and create train DataLoader

    train_dataset = NLPDataset('sst_train_raw.csv')
    text_freqs, label_freqs = build_frequency_dict(train_dataset)

    # Use token_freqs with the Vocab class
    text_vocab = Vocab(text_freqs, 'text', max_size=-1, min_freq=1)
    label_vocab = Vocab(label_freqs, 'label', max_size=-1, min_freq=1)
    train_dataset = NLPDataset('sst_train_raw.csv',text_vocab,label_vocab)


    glove_embeddings = load_glove_embeddings('sst_glove_6b_300d.txt')
    embedding_matrix = create_embedding_matrix(text_vocab, glove_embeddings)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=True)



    batch_size = 10
    shuffle = True
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi["<PAD>"]))


    # Load validation dataset and create validation DataLoader
    valid_dataset = NLPDataset('sst_valid_raw.csv',text_vocab,label_vocab)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi["<PAD>"]))
    
    # Load test dataset and create test DataLoader
    test_dataset = NLPDataset('sst_test_raw.csv',text_vocab,label_vocab)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi["<PAD>"]))

    # Initialize the baseline model
    # model = BaselineModel(embedding_layer, embedding_dim=300, hidden_dim=150, activation = 'relu', dropout_rate= 0)


    # Initialize the recurrent model, loss function, and optimizer
    model = RNNModel(embedding_layer, hidden_dim =150, num_layers=2, 
                     rnn_cell_type= 'lstm', dropout_rate=0, 
                     bidirectional=True, use_attention=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    gradient_clip = 1.0
    
    # Training and validation loop
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion, gradient_clip)
        avgLoss , accuracy , f1 , cm =evaluate(model, valid_dataloader, criterion)
        print(f"Epoch {epoch+1}: valid accuracy = {accuracy}  Average loss = {avgLoss}  f1_score = {f1}")

    # Test the model
    avgLoss , accuracy , f1 , cm =evaluate(model, test_dataloader, criterion)
    print(f"test accuracy = {accuracy}  Average loss = {avgLoss}  f1_score = {f1}")
    print("CM",cm)
if __name__ == "__main__":
    main()


"""
use_attention = False
seed = 705200
 LSTM
Epoch 1: valid accuracy = 0.785831960461285  Average loss = 0.4708855180629616  f1_score = 0.7732558139534882
Epoch 2: valid accuracy = 0.800658978583196  Average loss = 0.42573219680232427  f1_score = 0.8076311605723371
Epoch 3: valid accuracy = 0.8077979132344866  Average loss = 0.42418731510883473  f1_score = 0.8132337246531484
Epoch 4: valid accuracy = 0.7990115321252059  Average loss = 0.4290328850674499  f1_score = 0.8124999999999999
Epoch 5: valid accuracy = 0.800109829763866  Average loss = 0.43770687910632733  f1_score = 0.7975528364849833
test accuracy = 0.8027522935779816  Average loss = 0.43611760632219637  f1_score = 0.7995337995337994
CM [[357  87]
 [ 85 343]]

 

 use_attention = True
Epoch 1: valid accuracy = 0.7946183415705657  Average loss = 0.46099475188984895  f1_score = 0.7901234567901234
Epoch 2: valid accuracy = 0.7819879187259747  Average loss = 0.45712182312552396  f1_score = 0.7750708215297449
Epoch 3: valid accuracy = 0.8066996155958265  Average loss = 0.4235381699406384  f1_score = 0.8147368421052632
Epoch 4: valid accuracy = 0.7962657880285557  Average loss = 0.5223464078671946  f1_score = 0.7790351399642644
Epoch 5: valid accuracy = 0.8056013179571664  Average loss = 0.42683974262632307  f1_score = 0.8206686930091185
test accuracy = 0.7912844036697247  Average loss = 0.458208871582015  f1_score = 0.8038793103448276
CM [[317 127]
 [ 55 373]]

"""