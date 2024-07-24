"""
RNN cell comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loading import*

torch.manual_seed(8200)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers=2, rnn_cell_type='gru', dropout_rate=0.5, bidirectional=False):
        super(RNNModel, self).__init__()
        self.embedding = embedding_layer
        self.rnn_cell_type = rnn_cell_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Factor to multiply hidden_dim with, depending on bidirectionality
        factor = 2 if bidirectional else 1

        if rnn_cell_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_layer.embedding_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        elif rnn_cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_layer.embedding_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate if num_layers > 1 else 0,
                               bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_layer.embedding_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        else:
            raise ValueError("Invalid RNN cell type")

        self.fc1 = nn.Linear(self.hidden_dim * factor, self.hidden_dim)  
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # For bidirectional, use concatenated last hidden states of both directions
        if self.rnn_cell_type == 'lstm':# and self.bidirectional:
            # hidden = output[torch.arange(output.size(0)), lengths - 1, :self.hidden_dim] + output[torch.arange(output.size(0)), 0, self.hidden_dim:]
            tensor1 = output[torch.arange(output.size(0)), lengths - 1, :self.hidden_dim]
            tensor2 = output[torch.arange(output.size(0)), 0, self.hidden_dim:]
            hidden = torch.cat((tensor1, tensor2), dim=1)
        else:
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
    model = RNNModel(embedding_layer, hidden_dim =150, num_layers=2, rnn_cell_type= 'lstm', dropout_rate=0, bidirectional=True)
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
RNN cell comparison

NB: when changing a hyperparameter value only that hyperparameter is altered others remain at their default values.

The same hyperparameters (such as learning rate, optimizer, number of epoch, ...) as in the lab guide is used:

seed = 7052020
 GRU
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]

 LSTM
Epoch 1: valid accuracy = 0.7671609006040637
Epoch 2: valid accuracy = 0.7781438769906645
Epoch 3: valid accuracy = 0.7935200439319056
Epoch 4: valid accuracy = 0.7726523887973641
Epoch 5: valid accuracy = 0.7896760021965953
test accuracy = 0.7855504587155964
CM [[301 143]
[ 44 384]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7803404722679846
Epoch 4: valid accuracy = 0.7748489840746843
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7901376146788991
CM [[363  81]
 [102 326]]

seed = 705200
 GRU
Epoch 1: valid accuracy = 0.7742998352553542
Epoch 2: valid accuracy = 0.7578253706754531
Epoch 3: valid accuracy = 0.7957166392092258
Epoch 4: valid accuracy = 0.7874794069192751
Epoch 5: valid accuracy = 0.7814387699066447
test accuracy = 0.7958715596330275
CM [[401  43]
 [135 293]]

 LSTM
Epoch 1: valid accuracy = 0.7792421746293245
Epoch 2: valid accuracy = 0.7896760021965953
Epoch 3: valid accuracy = 0.7891268533772653
Epoch 4: valid accuracy = 0.8061504667764964
Epoch 5: valid accuracy = 0.7825370675453048
test accuracy = 0.7706422018348624
CM [[289 155]
 [ 45 383]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7655134541460736
Epoch 2: valid accuracy = 0.7506864360241625
Epoch 3: valid accuracy = 0.7748489840746843
Epoch 4: valid accuracy = 0.786381109280615
Epoch 5: valid accuracy = 0.7935200439319056
test accuracy = 0.7924311926605505
CM [[357  87]
 [ 94 334]]

seed  = 15200
 GRU
Epoch 1: valid accuracy = 0.7830862163646348
Epoch 2: valid accuracy = 0.7753981328940143
Epoch 3: valid accuracy = 0.7841845140032949
Epoch 4: valid accuracy = 0.7677100494233937
Epoch 5: valid accuracy = 0.7957166392092258
test accuracy = 0.8004587155963303
CM [[325 119]
 [ 55 373]]

 LSTM
Epoch 1: valid accuracy = 0.7753981328940143
Epoch 2: valid accuracy = 0.7913234486545854
Epoch 3: valid accuracy = 0.7655134541460736
Epoch 4: valid accuracy = 0.7995606809445359
Epoch 5: valid accuracy = 0.7973640856672158
test accuracy = 0.8073394495412844
CM [[359  85]
 [ 83 345]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7501372872048325
Epoch 2: valid accuracy = 0.7468423942888522
Epoch 3: valid accuracy = 0.7918725974739155
Epoch 4: valid accuracy = 0.785282811641955
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7786697247706422
CM [[342 102]
 [ 91 337]]


 *******************************************************************************************************************
 Effect of hidden_size
 
 seed = 7052020
 hidden_size = 150
 GRU
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]

  LSTM
Epoch 1: valid accuracy = 0.7671609006040637
Epoch 2: valid accuracy = 0.7781438769906645
Epoch 3: valid accuracy = 0.7935200439319056
Epoch 4: valid accuracy = 0.7726523887973641
Epoch 5: valid accuracy = 0.7896760021965953
test accuracy = 0.7855504587155964
CM [[301 143]
[ 44 384]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7803404722679846
Epoch 4: valid accuracy = 0.7748489840746843
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7901376146788991
CM [[363  81]
 [102 326]]

 hidden_size = 50
 GRU
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7726523887973641
Epoch 3: valid accuracy = 0.785282811641955
Epoch 4: valid accuracy = 0.7732015376166941
Epoch 5: valid accuracy = 0.785831960461285
test accuracy = 0.7878440366972477
CM [[376  68]
 [117 311]]

 LSTM
Epoch 1: valid accuracy = 0.7545304777594728
Epoch 2: valid accuracy = 0.7742998352553542
Epoch 3: valid accuracy = 0.7528830313014827
Epoch 4: valid accuracy = 0.7902251510159253
Epoch 5: valid accuracy = 0.7973640856672158
test accuracy = 0.7993119266055045
CM [[358  86]
 [ 89 339]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.6886326194398682
Epoch 2: valid accuracy = 0.757276221856123
Epoch 3: valid accuracy = 0.7688083470620538
Epoch 4: valid accuracy = 0.7803404722679846
Epoch 5: valid accuracy = 0.7869302580999451
test accuracy = 0.7729357798165137
CM [[351  93]
 [105 323]]

hidden_size = 300
GRU
Epoch 1: valid accuracy = 0.7792421746293245
Epoch 2: valid accuracy = 0.7759472817133443
Epoch 3: valid accuracy = 0.8012081274025261
Epoch 4: valid accuracy = 0.7929708951125755
Epoch 5: valid accuracy = 0.8066996155958265
test accuracy = 0.8096330275229358
CM [[340 104]
 [ 62 366]]

 LSTM
Epoch 1: valid accuracy = 0.7270730367929709
Epoch 2: valid accuracy = 0.7578253706754531
Epoch 3: valid accuracy = 0.800658978583196
Epoch 4: valid accuracy = 0.8034047226798462
Epoch 5: valid accuracy = 0.8017572762218561
test accuracy = 0.8153669724770642
CM [[347  97]
 [ 64 364]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7688083470620538
Epoch 2: valid accuracy = 0.7808896210873146
Epoch 3: valid accuracy = 0.7666117517847336
Epoch 4: valid accuracy = 0.7699066447007139
Epoch 5: valid accuracy = 0.7633168588687534
test accuracy = 0.7557339449541285
CM [[291 153]
 [ 60 368]]

*****************************************************************************************************************************
 Effect of num_layers
 
 seed = 7052020
 num_layers = 2
 GRU
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]

 LSTM
Epoch 1: valid accuracy = 0.7671609006040637
Epoch 2: valid accuracy = 0.7781438769906645
Epoch 3: valid accuracy = 0.7935200439319056
Epoch 4: valid accuracy = 0.7726523887973641
Epoch 5: valid accuracy = 0.7896760021965953
test accuracy = 0.7855504587155964
CM [[301 143]
[ 44 384]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7803404722679846
Epoch 4: valid accuracy = 0.7748489840746843
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7901376146788991
CM [[363  81]
 [102 326]]

 num_layers = 1
 GRU
Epoch 1: valid accuracy = 0.7737506864360242
Epoch 2: valid accuracy = 0.7891268533772653
Epoch 3: valid accuracy = 0.7990115321252059
Epoch 4: valid accuracy = 0.800658978583196
Epoch 5: valid accuracy = 0.7781438769906645
test accuracy = 0.7901376146788991
CM [[393  51]
 [132 296]]

 LSTM
Epoch 1: valid accuracy = 0.7693574958813838
Epoch 2: valid accuracy = 0.8023064250411862
Epoch 3: valid accuracy = 0.7699066447007139
Epoch 4: valid accuracy = 0.786381109280615
Epoch 5: valid accuracy = 0.8028555738605162
test accuracy = 0.8027522935779816
CM [[368  76]
 [ 96 332]]

  Vanilla RNN
Epoch 1: valid accuracy = 0.742998352553542
Epoch 2: valid accuracy = 0.7786930258099946
Epoch 3: valid accuracy = 0.7440966501922021
Epoch 4: valid accuracy = 0.7841845140032949
Epoch 5: valid accuracy = 0.7808896210873146
test accuracy = 0.7637614678899083
CM [[310 134]
 [ 72 356]]



 num_layers = 5
 GRU
Epoch 1: valid accuracy = 0.7561779242174629
Epoch 2: valid accuracy = 0.7792421746293245
Epoch 3: valid accuracy = 0.7682591982427238
Epoch 4: valid accuracy = 0.8023064250411862
Epoch 5: valid accuracy = 0.7990115321252059
test accuracy = 0.8027522935779816
CM [[382  62]
 [110 318]]

 LSTM
Epoch 1: valid accuracy = 0.7545304777594728
Epoch 2: valid accuracy = 0.7781438769906645
Epoch 3: valid accuracy = 0.7721032399780341
Epoch 4: valid accuracy = 0.7918725974739155
Epoch 5: valid accuracy = 0.7946183415705657
test accuracy = 0.7981651376146789
CM [[342 102]
 [ 74 354]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7561779242174629
Epoch 2: valid accuracy = 0.742998352553542
Epoch 3: valid accuracy = 0.7649643053267435
Epoch 4: valid accuracy = 0.7781438769906645
Epoch 5: valid accuracy = 0.7616694124107634
test accuracy = 0.7752293577981652
CM [[295 149]
 [ 47 381]]

 *************************************************************************************************************************************
 Effect of dropout
 
 seed = 7052020
 dropout = 0
 GRU
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]

  LSTM
Epoch 1: valid accuracy = 0.7671609006040637
Epoch 2: valid accuracy = 0.7781438769906645
Epoch 3: valid accuracy = 0.7935200439319056
Epoch 4: valid accuracy = 0.7726523887973641
Epoch 5: valid accuracy = 0.7896760021965953
test accuracy = 0.7855504587155964
CM [[301 143]
[ 44 384]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7803404722679846
Epoch 4: valid accuracy = 0.7748489840746843
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7901376146788991
CM [[363  81]
 [102 326]]



 dropout = 0.4
 GRU
Epoch 1: valid accuracy = 0.7649643053267435
Epoch 2: valid accuracy = 0.7748489840746843
Epoch 3: valid accuracy = 0.7913234486545854
Epoch 4: valid accuracy = 0.8066996155958265
Epoch 5: valid accuracy = 0.8066996155958265
test accuracy = 0.7993119266055045
CM [[342 102]
 [ 73 355]]

 LSTM
 Epoch 1: valid accuracy = 0.771004942339374
Epoch 2: valid accuracy = 0.7742998352553542
Epoch 3: valid accuracy = 0.7808896210873146
Epoch 4: valid accuracy = 0.8012081274025261
Epoch 5: valid accuracy = 0.8028555738605162
test accuracy = 0.8027522935779816
CM [[340 104]
 [ 68 360]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7627677100494233
Epoch 2: valid accuracy = 0.7814387699066447
Epoch 3: valid accuracy = 0.7830862163646348
Epoch 4: valid accuracy = 0.771554091158704
Epoch 5: valid accuracy = 0.7907742998352554
test accuracy = 0.7844036697247706
CM [[368  76]
 [112 316]]

 dropout = 0.8
 GRU
Epoch 1: valid accuracy = 0.7578253706754531
Epoch 2: valid accuracy = 0.771004942339374
Epoch 3: valid accuracy = 0.7918725974739155
Epoch 4: valid accuracy = 0.7973640856672158
Epoch 5: valid accuracy = 0.800109829763866
test accuracy = 0.7947247706422018
CM [[325 119]
 [ 60 368]]

 LSTM
Epoch 1: valid accuracy = 0.7644151565074135
Epoch 2: valid accuracy = 0.7512355848434926
Epoch 3: valid accuracy = 0.785282811641955
Epoch 4: valid accuracy = 0.8023064250411862
Epoch 5: valid accuracy = 0.7995606809445359
test accuracy = 0.8061926605504587
CM [[346  98]
 [ 71 357]]

 Vanilla RNN
 Epoch 1: valid accuracy = 0.7364085667215815
Epoch 2: valid accuracy = 0.7677100494233937
Epoch 3: valid accuracy = 0.7688083470620538
Epoch 4: valid accuracy = 0.7808896210873146
Epoch 5: valid accuracy = 0.7830862163646348
test accuracy = 0.768348623853211
CM [[367  77]
 [125 303]]

 
 dropout = 1
 GRU
Epoch 1: valid accuracy = 0.49917627677100496
Epoch 2: valid accuracy = 0.49917627677100496
Epoch 3: valid accuracy = 0.49917627677100496
Epoch 4: valid accuracy = 0.499725425590335
Epoch 5: valid accuracy = 0.49917627677100496
test accuracy = 0.5091743119266054
CM [[444   0]
 [428   0]]

 LSTM
Epoch 1: valid accuracy = 0.49917627677100496
Epoch 2: valid accuracy = 0.499725425590335
Epoch 3: valid accuracy = 0.49917627677100496
Epoch 4: valid accuracy = 0.4986271279516749
Epoch 5: valid accuracy = 0.49917627677100496
test accuracy = 0.5091743119266054
CM [[444   0]
 [428   0]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.5041186161449753
Epoch 2: valid accuracy = 0.5189456342668863
Epoch 3: valid accuracy = 0.5068643602416255
Epoch 4: valid accuracy = 0.4986271279516749
Epoch 5: valid accuracy = 0.5118066996155958
test accuracy = 0.4919724770642202
CM [[381  63]
 [380  48]]


 ********************************************************************************************************************************
 Effect of bidirectional
 
 seed = 7052020
 bidirectional = False
 GRU
Epoch 1: valid accuracy = 0.7770455793520044
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.8039538714991763
Epoch 4: valid accuracy = 0.7990115321252059
Epoch 5: valid accuracy = 0.7951674903898956
test accuracy = 0.7912844036697247
CM [[329 115]
 [ 67 361]]

 LSTM
Epoch 1: valid accuracy = 0.7753981328940143
Epoch 2: valid accuracy = 0.7913234486545854
Epoch 3: valid accuracy = 0.7655134541460736
Epoch 4: valid accuracy = 0.7995606809445359
Epoch 5: valid accuracy = 0.7973640856672158
test accuracy = 0.8073394495412844
CM [[359  85]
 [ 83 345]]

 Vanilla RNN
Epoch 1: valid accuracy = 0.7347611202635914
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7803404722679846
Epoch 4: valid accuracy = 0.7748489840746843
Epoch 5: valid accuracy = 0.7775947281713345
test accuracy = 0.7901376146788991
CM [[363  81]
 [102 326]]

 bidirectional = True
 GRU
Epoch 1: valid accuracy = 0.7775947281713345
Epoch 2: valid accuracy = 0.7819879187259747
Epoch 3: valid accuracy = 0.7962657880285557
Epoch 4: valid accuracy = 0.7929708951125755
Epoch 5: valid accuracy = 0.8138385502471169
test accuracy = 0.8061926605504587
CM [[359  85]
 [ 84 344]]

 LSTM
Epoch 1: valid accuracy = 0.771554091158704
Epoch 2: valid accuracy = 0.7770455793520044
Epoch 3: valid accuracy = 0.785282811641955
Epoch 4: valid accuracy = 0.8023064250411862
Epoch 5: valid accuracy = 0.7979132344865458
test accuracy = 0.819954128440367
CM [[382  62]
 [ 95 333]]

 Vanilla RNN
 Epoch 1: valid accuracy = 0.7732015376166941
Epoch 2: valid accuracy = 0.7764964305326744
Epoch 3: valid accuracy = 0.7880285557386052
Epoch 4: valid accuracy = 0.7918725974739155
Epoch 5: valid accuracy = 0.7742998352553542
test accuracy = 0.7786697247706422
CM [[369  75]
 [118 310]]

***********************************************************************************************************************************

 THE BEST MODEL

 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True

seed = 7052020
Epoch 1: valid accuracy = 0.7841845140032949
Epoch 2: valid accuracy = 0.800109829763866
Epoch 3: valid accuracy = 0.7962657880285557
Epoch 4: valid accuracy = 0.8039538714991763
Epoch 5: valid accuracy = 0.7990115321252059
test accuracy = 0.8107798165137615
CM [[389  55]
 [110 318]]

seed = 705200
Epoch 1: valid accuracy = 0.7836353651839648
Epoch 2: valid accuracy = 0.7984623833058759
Epoch 3: valid accuracy = 0.800658978583196
Epoch 4: valid accuracy = 0.7973640856672158
Epoch 5: valid accuracy = 0.8039538714991763
test accuracy = 0.8142201834862385
CM [[357  87]
 [ 75 353]]

seed  = 15200
Epoch 1: valid accuracy = 0.771554091158704
Epoch 2: valid accuracy = 0.7770455793520044
Epoch 3: valid accuracy = 0.785282811641955
Epoch 4: valid accuracy = 0.8023064250411862
Epoch 5: valid accuracy = 0.7979132344865458
test accuracy = 0.819954128440367
CM [[382  62]
 [ 95 333]]

seed = 158200
Epoch 1: valid accuracy = 0.7902251510159253
Epoch 2: valid accuracy = 0.7924217462932455
Epoch 3: valid accuracy = 0.800109829763866
Epoch 4: valid accuracy = 0.8012081274025261
Epoch 5: valid accuracy = 0.7918725974739155
test accuracy = 0.7809633027522935
CM [[301 143]
 [ 48 380]]

seed = 8200
Epoch 1: valid accuracy = 0.7781438769906645
Epoch 2: valid accuracy = 0.7984623833058759
Epoch 3: valid accuracy = 0.7935200439319056
Epoch 4: valid accuracy = 0.8105436573311368
Epoch 5: valid accuracy = 0.7830862163646348
test accuracy = 0.7935779816513762
CM [[398  46]
 [134 294]]

"""