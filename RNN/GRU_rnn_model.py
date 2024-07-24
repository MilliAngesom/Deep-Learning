import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loading import*

torch.manual_seed(8200)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers=2):
        super(RNNModel, self).__init__()
        self.embedding = embedding_layer
        self.rnn = nn.GRU(input_size=embedding_layer.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    

    def forward(self, x, lengths):
        x = self.embedding(x)
        # Packing the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_x)
        # Optionally, we can unpack the output, or directly use it for further processing
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Extracting the last relevant hidden states for classification
        hidden = output[torch.arange(output.size(0)), lengths - 1]
        hidden = self.relu(self.fc1(hidden))
        output = self.fc2(hidden)
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
    text_vocab = Vocab(text_freqs, 'text', max_size=-1, min_freq=0)
    label_vocab = Vocab(label_freqs, 'label', max_size=-1, min_freq=0)
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

    # Initialize model, loss function, and optimizer
    model = RNNModel(embedding_layer,  hidden_dim=150)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    gradient_clip = 0.25
    
    # Training and validation loop
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion, gradient_clip)
        _ , accuracy , _ , cm =evaluate(model, valid_dataloader, criterion)
        print(f"Epoch {epoch+1}: valid accuracy = {accuracy}")

    # Test the model
    _ , accuracy , _ , cm =evaluate(model, test_dataloader, criterion)
    print(f"test accuracy = {accuracy}")
    print("CM",cm)
if __name__ == "__main__":
    main()


"""
The same hyperparameters as in the lab guide is used:

seed = 7052020
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]


seed = 705200
Epoch 1: valid accuracy = 0.7742998352553542
Epoch 2: valid accuracy = 0.7578253706754531
Epoch 3: valid accuracy = 0.7957166392092258
Epoch 4: valid accuracy = 0.7874794069192751
Epoch 5: valid accuracy = 0.7814387699066447
test accuracy = 0.7958715596330275
CM [[401  43]
 [135 293]]


seed  = 15200
Epoch 1: valid accuracy = 0.7830862163646348
Epoch 2: valid accuracy = 0.7753981328940143
Epoch 3: valid accuracy = 0.7841845140032949
Epoch 4: valid accuracy = 0.7677100494233937
Epoch 5: valid accuracy = 0.7957166392092258
test accuracy = 0.8004587155963303
CM [[325 119]
 [ 55 373]]


seed = 158200
Epoch 1: valid accuracy = 0.7594728171334432
Epoch 2: valid accuracy = 0.7902251510159253
Epoch 3: valid accuracy = 0.7918725974739155
Epoch 4: valid accuracy = 0.7979132344865458
Epoch 5: valid accuracy = 0.7935200439319056
test accuracy = 0.8084862385321101
CM [[369  75]
 [ 92 336]]


seed = 8200
Epoch 1: valid accuracy = 0.7764964305326744
Epoch 2: valid accuracy = 0.7869302580999451
Epoch 3: valid accuracy = 0.800658978583196
Epoch 4: valid accuracy = 0.800658978583196
Epoch 5: valid accuracy = 0.7891268533772653
test accuracy = 0.7993119266055045
CM [[397  47]
 [128 300]]
"""