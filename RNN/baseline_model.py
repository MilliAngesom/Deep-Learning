import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loading import*

torch.manual_seed(8200)

# model
class BaselineModel(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, hidden_dim, activation='relu', dropout_rate=0.5):
        super(BaselineModel, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid Activation Function")

    def forward(self, x, lengths):
        x = self.embedding(x)
        sum_embeddings = torch.sum(x, dim=1)
        lengths = lengths.view(-1, 1).to(sum_embeddings.dtype)
        mean_embeddings = sum_embeddings / lengths

        hidden = self.activation(self.fc1(mean_embeddings))
        hidden = self.dropout(hidden)  
        hidden = self.activation(self.fc2(hidden))
        output = self.fc3(hidden)
        return output





#training loop
def train(model, data_loader, optimizer, criterion):
    model.train()
    for texts, labels,lengths in data_loader:
        optimizer.zero_grad()
        labels = labels.view(-1, 1)
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels.float())
        loss.backward()
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
    model = BaselineModel(embedding_layer, embedding_dim=300, hidden_dim=150)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    
    # Training and validation loop
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion)
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
Epoch 1: valid accuracy = 0.7655134541460736
Epoch 2: valid accuracy = 0.7770455793520044
Epoch 3: valid accuracy = 0.7929708951125755
Epoch 4: valid accuracy = 0.7841845140032949
Epoch 5: valid accuracy = 0.7940691927512356
test accuracy = 0.7775229357798165
CM [[346  98]
 [ 96 332]]


seed = 705200
Epoch 1: valid accuracy = 0.7704557935200439
Epoch 2: valid accuracy = 0.7896760021965953
Epoch 3: valid accuracy = 0.785831960461285
Epoch 4: valid accuracy = 0.7737506864360242
Epoch 5: valid accuracy = 0.7940691927512356
test accuracy = 0.7844036697247706
CM [[371  73]
 [115 313]]


seed  = 15200
Epoch 1: valid accuracy = 0.7583745194947831
Epoch 2: valid accuracy = 0.7786930258099946
Epoch 3: valid accuracy = 0.7759472817133443
Epoch 4: valid accuracy = 0.7786930258099946
Epoch 5: valid accuracy = 0.7803404722679846
test accuracy = 0.7729357798165137
CM [[381  63]
 [135 293]]


seed = 158200
Epoch 1: valid accuracy = 0.7825370675453048
Epoch 2: valid accuracy = 0.7929708951125755
Epoch 3: valid accuracy = 0.786381109280615
Epoch 4: valid accuracy = 0.7957166392092258
Epoch 5: valid accuracy = 0.7979132344865458
test accuracy = 0.7798165137614679
CM [[355  89]
 [103 325]]


seed = 8200
Epoch 1: valid accuracy = 0.7682591982427238
Epoch 2: valid accuracy = 0.7841845140032949
Epoch 3: valid accuracy = 0.786381109280615
Epoch 4: valid accuracy = 0.786381109280615
Epoch 5: valid accuracy = 0.7880285557386052
test accuracy = 0.7729357798165137
CM [[331 113]
 [ 85 343]]
"""