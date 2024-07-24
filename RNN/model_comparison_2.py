"""
Hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loading import*
from baseline_model import*

torch.manual_seed(8200)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers=2, rnn_cell_type='gru', dropout_rate=0.5, bidirectional=False, activation = 'relu'):
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

        self.fc1 = nn.Linear(self.hidden_dim * factor, self.hidden_dim)  # considering bidirectional
        self.fc2 = nn.Linear(self.hidden_dim, 1)
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
        hidden = self.activation(self.fc1(hidden))
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
    model = RNNModel(embedding_layer, hidden_dim =150, num_layers=2, rnn_cell_type= 'lstm', dropout_rate=0, bidirectional=True, activation = 'relu')
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
Hyperparameter optimization

NB: when changing a hyperparameter value only that hyperparameter is altered others remain at their default values.

The same hyperparameters (such as learning rate, optimizer, number of epoch, ...) as in the lab guide is used:

***********************************************************************************************************************************************
Using pretrained vector representations:

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]

 Without pretrained vector representations:

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.49917627677100496  Average loss = 0.6937780191338128  f1_score = 0.0
Epoch 2: valid accuracy = 0.49917627677100496  Average loss = 0.6924562288112328  f1_score = 0.0
Epoch 3: valid accuracy = 0.7660626029654036  Average loss = 0.4975115726880037  f1_score = 0.7753164556962026
Epoch 4: valid accuracy = 0.7682591982427238  Average loss = 0.7930334539768474  f1_score = 0.8035381750465549
Epoch 5: valid accuracy = 0.7896760021965953  Average loss = 0.8420830896007192  f1_score = 0.7685800604229607
test accuracy = 0.7752293577981652  Average loss = 0.9319162143987011  f1_score = 0.7487179487179487
CM [[384  60]
 [136 292]]


 Baseline Model


Epoch 1: valid accuracy = 0.49917627677100496  Average loss = 0.6939872848531587  f1_score = 0.0
Epoch 2: valid accuracy = 0.49917627677100496  Average loss = 0.6955827500650792  f1_score = 0.0
Epoch 3: valid accuracy = 0.6529379461834157  Average loss = 0.6071877284128158  f1_score = 0.5204855842185129
Epoch 4: valid accuracy = 0.7649643053267435  Average loss = 0.48683359763009953  f1_score = 0.76353591160221
Epoch 5: valid accuracy = 0.7913234486545854  Average loss = 0.4465681084436797  f1_score = 0.8039215686274509
test accuracy = 0.7798165137614679  Average loss = 0.45965105202049017  f1_score = 0.7876106194690267
CM [[324 120]
 [ 72 356]]



**********************************************************************************************************************************************



Effect of Vocabulary size

Vocabulary size = 14806 (all available words)

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]

 
 Vocabulary size = 1000

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7215815485996705  Average loss = 0.558211118471427  f1_score = 0.7525622254758418
Epoch 2: valid accuracy = 0.7160900604063701  Average loss = 0.5504288181988268  f1_score = 0.6860959319975712
Epoch 3: valid accuracy = 0.7375068643602416  Average loss = 0.5162395906578647  f1_score = 0.7492130115424974
Epoch 4: valid accuracy = 0.7402526084568918  Average loss = 0.5161921262252526  f1_score = 0.7447382622773879
Epoch 5: valid accuracy = 0.7166392092257001  Average loss = 0.5467387867246658  f1_score = 0.7669376693766938
test accuracy = 0.7190366972477065  Average loss = 0.546730110083114  f1_score = 0.7659980897803248
CM [[226 218]
 [ 27 401]]

 Baseline Model

 Epoch 1: valid accuracy = 0.6639209225700164  Average loss = 0.6303616778446677  f1_score = 0.6126582278481013
Epoch 2: valid accuracy = 0.6661175178473366  Average loss = 0.5929211843860606  f1_score = 0.5777777777777777
Epoch 3: valid accuracy = 0.7155409115870401  Average loss = 0.5525674393268231  f1_score = 0.6802469135802469
Epoch 4: valid accuracy = 0.7210323997803405  Average loss = 0.5409587426263778  f1_score = 0.6994082840236686
Epoch 5: valid accuracy = 0.7265238879736409  Average loss = 0.5318967146300227  f1_score = 0.710128055878929
test accuracy = 0.7511467889908257  Average loss = 0.5209306195716966  f1_score = 0.7330873308733087
CM [[357  87]
 [130 298]]


  Vocabulary size = 100

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.5881383855024712  Average loss = 0.669585496838627  f1_score = 0.5324189526184538
Epoch 2: valid accuracy = 0.6128500823723229  Average loss = 0.6506756226547429  f1_score = 0.6401225114854519
Epoch 3: valid accuracy = 0.642504118616145  Average loss = 0.6363362946796939  f1_score = 0.6186291739894553
Epoch 4: valid accuracy = 0.6370126304228446  Average loss = 0.6329930484946308  f1_score = 0.5957186544342508
Epoch 5: valid accuracy = 0.6518396485447556  Average loss = 0.621897381633683  f1_score = 0.6852035749751738
test accuracy = 0.661697247706422  Average loss = 0.6155834651806138  f1_score = 0.6917450365726227
CM [[246 198]
 [ 97 331]]

  Baseline Model

Epoch 1: valid accuracy = 0.5507962657880285  Average loss = 0.6864882124577715  f1_score = 0.336038961038961
Epoch 2: valid accuracy = 0.5584843492586491  Average loss = 0.6802987407465451  f1_score = 0.34846029173419774
Epoch 3: valid accuracy = 0.5892366831411312  Average loss = 0.6680221345906701  f1_score = 0.46799431009957326
Epoch 4: valid accuracy = 0.6018671059857221  Average loss = 0.6594905198597517  f1_score = 0.5156980627922512
Epoch 5: valid accuracy = 0.5760571114772103  Average loss = 0.6668289870512291  f1_score = 0.37842190016103056
test accuracy = 0.6135321100917431  Average loss = 0.6546445299278606  f1_score = 0.43170320404721757
CM [[407  37]
 [300 128]]


****************************************************************************************************************************************************

Effect of Batch size

Batch size = 10

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]

 

 Batch size = 100

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.6523887973640856  Average loss = 0.6612064399217304  f1_score = 0.5449317038102085
Epoch 2: valid accuracy = 0.7605711147721033  Average loss = 0.4838954178910506  f1_score = 0.7817817817817817
Epoch 3: valid accuracy = 0.7929708951125755  Average loss = 0.4654246035375093  f1_score = 0.7961060032449973
Epoch 4: valid accuracy = 0.7759472817133443  Average loss = 0.46852949889082657  f1_score = 0.790983606557377
Epoch 5: valid accuracy = 0.7924217462932455  Average loss = 0.4530696398333499  f1_score = 0.78125
test accuracy = 0.7947247706422018  Average loss = 0.4562539325820075  f1_score = 0.7825030376670717
CM [[371  73]
 [106 322]]

 Baseline Model
Epoch 1: valid accuracy = 0.5507962657880285  Average loss = 0.6796847644605135  f1_score = 0.22684310018903592
Epoch 2: valid accuracy = 0.7237781438769907  Average loss = 0.6299325447333487  f1_score = 0.7046388725778039
Epoch 3: valid accuracy = 0.7616694124107634  Average loss = 0.555664460909994  f1_score = 0.7528473804100229
Epoch 4: valid accuracy = 0.7666117517847336  Average loss = 0.5023217703166761  f1_score = 0.785245073269328
Epoch 5: valid accuracy = 0.785282811641955  Average loss = 0.4678093932176891  f1_score = 0.7885343428880476
test accuracy = 0.7729357798165137  Average loss = 0.47277260488933986  f1_score = 0.7692307692307693
CM [[344 100]
 [ 98 330]]


 Batch size = 1

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.5507962657880285  Average loss = 0.6796847644605135  f1_score = 0.22684310018903592
Epoch 2: valid accuracy = 0.7237781438769907  Average loss = 0.6299325447333487  f1_score = 0.7046388725778039
Epoch 3: valid accuracy = 0.7616694124107634  Average loss = 0.555664460909994  f1_score = 0.7528473804100229
Epoch 4: valid accuracy = 0.7666117517847336  Average loss = 0.5023217703166761  f1_score = 0.785245073269328
Epoch 5: valid accuracy = 0.785282811641955  Average loss = 0.4678093932176891  f1_score = 0.7885343428880476
test accuracy = 0.7729357798165137  Average loss = 0.47277260488933986  f1_score = 0.7692307692307693
CM [[344 100]
 [ 98 330]]

 Baseline Model

Epoch 1: valid accuracy = 0.7759472817133443  Average loss = 0.7188160088651023  f1_score = 0.7920489296636086
Epoch 2: valid accuracy = 0.7891268533772653  Average loss = 0.7753295365281927  f1_score = 0.7873754152823921
Epoch 3: valid accuracy = 0.7907742998352554  Average loss = 0.8302501749504612  f1_score = 0.7821612349914236
Epoch 4: valid accuracy = 0.7968149368478857  Average loss = 0.8601665102057038  f1_score = 0.7930648769574945
Epoch 5: valid accuracy = 0.7786930258099946  Average loss = 0.9568025297112265  f1_score = 0.7983991995997999
test accuracy = 0.7763761467889908  Average loss = 1.0382933798982232  f1_score = 0.7958115183246072
CM [[297 147]
 [ 48 380]]

 
 *********************************************************************************************************************************************


 
 Effect of Activation function

 Activation function = ReLU

  THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]


 Activation function = Sigmoid

  THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4757287576387489  f1_score = 0.7952058363731109
Epoch 2: valid accuracy = 0.8034047226798462  Average loss = 0.44267976705412393  f1_score = 0.8054347826086958
Epoch 3: valid accuracy = 0.800109829763866  Average loss = 0.4517953665774377  f1_score = 0.8063829787234043
Epoch 4: valid accuracy = 0.7913234486545854  Average loss = 0.447631800150285  f1_score = 0.7939262472885033
Epoch 5: valid accuracy = 0.8066996155958265  Average loss = 0.44414669333300627  f1_score = 0.805094130675526
test accuracy = 0.8130733944954128  Average loss = 0.4395127677443353  f1_score = 0.81068524970964
CM [[360  84]
 [ 79 349]]

 
 Baseline Model

Epoch 1: valid accuracy = 0.5387149917627677  Average loss = 0.6815107630901649  f1_score = 0.17485265225933203
Epoch 2: valid accuracy = 0.7018121911037891  Average loss = 0.6233399067420126  f1_score = 0.6521460602178091
Epoch 3: valid accuracy = 0.7523338824821527  Average loss = 0.5376737386476799  f1_score = 0.75178866263071
Epoch 4: valid accuracy = 0.771554091158704  Average loss = 0.49364433987218825  f1_score = 0.7911646586345382
Epoch 5: valid accuracy = 0.7874794069192751  Average loss = 0.46574568325053145  f1_score = 0.7865416436845009
test accuracy = 0.7729357798165137  Average loss = 0.47686759890480473  f1_score = 0.7648456057007127
CM [[352  92]
 [106 322]]


 Activation function = Tanh

  THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7770455793520044  Average loss = 0.45367829501628876  f1_score = 0.7930682976554535
Epoch 2: valid accuracy = 0.7984623833058759  Average loss = 0.4317940349176608  f1_score = 0.7948574622694242
Epoch 3: valid accuracy = 0.800109829763866  Average loss = 0.4369583001635114  f1_score = 0.8047210300429184
Epoch 4: valid accuracy = 0.7995606809445359  Average loss = 0.45649478214034617  f1_score = 0.7973348139922265
Epoch 5: valid accuracy = 0.7951674903898956  Average loss = 0.4853380913579012  f1_score = 0.7710251688152242
test accuracy = 0.8061926605504587  Average loss = 0.4789934290826998  f1_score = 0.7836107554417414
CM [[397  47]
 [122 306]]

 
 Baseline Model

Epoch 1: valid accuracy = 0.7748489840746843  Average loss = 0.4767879877096968  f1_score = 0.7807486631016042
Epoch 2: valid accuracy = 0.7797913234486545  Average loss = 0.45231685254091775  f1_score = 0.7886136004217186
Epoch 3: valid accuracy = 0.7935200439319056  Average loss = 0.4434468744963896  f1_score = 0.7868480725623582
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44332316872037825  f1_score = 0.7953216374269005
Epoch 5: valid accuracy = 0.7924217462932455  Average loss = 0.45225851791478244  f1_score = 0.78125
test accuracy = 0.7798165137614679  Average loss = 0.4743634496222843  f1_score = 0.7658536585365853
CM [[366  78]
 [114 314]]


************************************************************************************************************************************


Effect of Gradient clipping value

Gradient clipping value = 0.25

THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]


 Gradient clipping value = 1.0

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7902251510159253  Average loss = 0.44907253418789533  f1_score = 0.7932900432900432
Epoch 2: valid accuracy = 0.8034047226798462  Average loss = 0.4296200981597757  f1_score = 0.7997762863534676
Epoch 3: valid accuracy = 0.7995606809445359  Average loss = 0.4327748902752751  f1_score = 0.8085998951232303
Epoch 4: valid accuracy = 0.8061504667764964  Average loss = 0.442191241787431  f1_score = 0.8050800662617338
Epoch 5: valid accuracy = 0.8039538714991763  Average loss = 0.4551765984814879  f1_score = 0.7871198568872987
test accuracy = 0.8142201834862385  Average loss = 0.44671537075191736  f1_score = 0.7999999999999999
CM [[386  58]
 [104 324]]

 
 Baseline Model

Epoch 1: valid accuracy = 0.7737506864360242  Average loss = 0.49237295838653067  f1_score = 0.7799145299145299
Epoch 2: valid accuracy = 0.7841845140032949  Average loss = 0.44930278538354756  f1_score = 0.7846575342465755
Epoch 3: valid accuracy = 0.7918725974739155  Average loss = 0.43997239292970775  f1_score = 0.7876750700280113
Epoch 4: valid accuracy = 0.7891268533772653  Average loss = 0.4385859995591836  f1_score = 0.7944325481798715
Epoch 5: valid accuracy = 0.7885777045579352  Average loss = 0.44455261919342104  f1_score = 0.7783534830166954
test accuracy = 0.7763761467889908  Average loss = 0.46079697857864876  f1_score = 0.7642079806529625
CM [[361  83]
 [112 316]]


 Gradient clipping value = 10.0

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7913234486545854  Average loss = 0.44348576990633065  f1_score = 0.7970085470085471
Epoch 2: valid accuracy = 0.7885777045579352  Average loss = 0.4417114157860722  f1_score = 0.7712418300653595
Epoch 3: valid accuracy = 0.7918725974739155  Average loss = 0.4268706736506009  f1_score = 0.8035251425609123
Epoch 4: valid accuracy = 0.8077979132344866  Average loss = 0.4193764359898906  f1_score = 0.808743169398907
Epoch 5: valid accuracy = 0.8099945085118067  Average loss = 0.41313938406513  f1_score = 0.8060538116591928
test accuracy = 0.8142201834862385  Average loss = 0.42516945945945656  f1_score = 0.8111888111888111
CM [[362  82]
 [ 80 348]]


 Baseline Model

 Epoch 1: valid accuracy = 0.7655134541460736  Average loss = 0.4960319499174754  f1_score = 0.7804627249357327
Epoch 2: valid accuracy = 0.7770455793520044  Average loss = 0.46210196166416334  f1_score = 0.7568862275449103
Epoch 3: valid accuracy = 0.7929708951125755  Average loss = 0.44495987843294615  f1_score = 0.7842014882655981
Epoch 4: valid accuracy = 0.7841845140032949  Average loss = 0.4389807651114594  f1_score = 0.7921734531993654
Epoch 5: valid accuracy = 0.7940691927512356  Average loss = 0.43904382023003585  f1_score = 0.7947454844006568
test accuracy = 0.7775229357798165  Average loss = 0.4546659485521642  f1_score = 0.7738927738927739
CM [[346  98]
 [ 96 332]]


 Gradient clipping value = 0.05

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7775947281713345  Average loss = 0.4656460676444033  f1_score = 0.7858276044420941
Epoch 2: valid accuracy = 0.7951674903898956  Average loss = 0.44276857555238275  f1_score = 0.7903316469926926
Epoch 3: valid accuracy = 0.800658978583196  Average loss = 0.44441217717414344  f1_score = 0.8084432717678101
Epoch 4: valid accuracy = 0.800109829763866  Average loss = 0.4769494222080121  f1_score = 0.7912844036697247
Epoch 5: valid accuracy = 0.7825370675453048  Average loss = 0.48559546243982094  f1_score = 0.752808988764045
test accuracy = 0.7958715596330275  Average loss = 0.4748816926201636  f1_score = 0.7688311688311686
CM [[398  46]
 [132 296]]


 Baseline Model

Epoch 1: valid accuracy = 0.7677100494233937  Average loss = 0.5186404168931513  f1_score = 0.7727028479312197
Epoch 2: valid accuracy = 0.7781438769906645  Average loss = 0.4548106748061102  f1_score = 0.7857900318133616
Epoch 3: valid accuracy = 0.7896760021965953  Average loss = 0.44244497293820145  f1_score = 0.7837380011293056
Epoch 4: valid accuracy = 0.7918725974739155  Average loss = 0.4397683533651581  f1_score = 0.7967828418230564
Epoch 5: valid accuracy = 0.7896760021965953  Average loss = 0.4469891366411428  f1_score = 0.7802639127940332
test accuracy = 0.7786697247706422  Average loss = 0.4647503083741123  f1_score = 0.7649208282582217
CM [[365  79]
 [114 314]]


 ********************************************************************************************************************************************

 Effect of Dropout

Dropout = 0

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

Epoch 1: valid accuracy = 0.7841845140032949  Average loss = 0.4587690225227283  f1_score = 0.7930489731437599
Epoch 2: valid accuracy = 0.800109829763866  Average loss = 0.4353895378324503  f1_score = 0.8010928961748635
Epoch 3: valid accuracy = 0.7962657880285557  Average loss = 0.44129493704340494  f1_score = 0.8046340179041601
Epoch 4: valid accuracy = 0.8039538714991763  Average loss = 0.447425760573051  f1_score = 0.8050245767340252
Epoch 5: valid accuracy = 0.7990115321252059  Average loss = 0.4610730619106157  f1_score = 0.7792521109770809
test accuracy = 0.8107798165137615  Average loss = 0.45197082607244904  f1_score = 0.7940074906367042
CM [[389  55]
 [110 318]]

 Baseline Model

Epoch 1: valid accuracy = 0.7726523887973641  Average loss = 0.5015441630381704  f1_score = 0.7778969957081545
Epoch 2: valid accuracy = 0.7836353651839648  Average loss = 0.45286671214742086  f1_score = 0.7899786780383796
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.44246518905045557  f1_score = 0.7830508474576271
Epoch 4: valid accuracy = 0.7885777045579352  Average loss = 0.44044720108717517  f1_score = 0.7940074906367041
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4471615254064727  f1_score = 0.7812142038946163
test accuracy = 0.7763761467889908  Average loss = 0.46562289861454204  f1_score = 0.7630619684082626
CM [[363  81]
 [114 314]]

 

 Dropout = 0.5

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7847336628226249  Average loss = 0.4580495267536471  f1_score = 0.791044776119403
Epoch 2: valid accuracy = 0.7869302580999451  Average loss = 0.45605858244368286  f1_score = 0.7922912205567451
Epoch 3: valid accuracy = 0.7924217462932455  Average loss = 0.437403129729254  f1_score = 0.8043478260869564
Epoch 4: valid accuracy = 0.7671609006040637  Average loss = 0.5299437957042269  f1_score = 0.7302798982188295
Epoch 5: valid accuracy = 0.7896760021965953  Average loss = 0.4602725322821427  f1_score = 0.7807670291929021
test accuracy = 0.8027522935779816  Average loss = 0.45757606566290965  f1_score = 0.7932692307692307
CM [[370  74]
 [ 98 330]]


 Baseline Model

 Epoch 1: valid accuracy = 0.7413509060955519  Average loss = 0.584711879491806  f1_score = 0.7384786229872293
Epoch 2: valid accuracy = 0.7742998352553542  Average loss = 0.4817178540868186  f1_score = 0.7814992025518341
Epoch 3: valid accuracy = 0.7792421746293245  Average loss = 0.45229639874292854  f1_score = 0.7908428720083247
Epoch 4: valid accuracy = 0.7819879187259747  Average loss = 0.4430345017564753  f1_score = 0.7905013192612137
Epoch 5: valid accuracy = 0.7880285557386052  Average loss = 0.4394479094307279  f1_score = 0.7970557308096742
test accuracy = 0.783256880733945  Average loss = 0.45802222480150784  f1_score = 0.7869222096956031
CM [[334 110]
 [ 79 349]]



 Dropout = 0.8

 THE BEST MODEL{
 LSTM
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True }

 Epoch 1: valid accuracy = 0.7737506864360242  Average loss = 0.4689391276549772  f1_score = 0.7910750507099391
Epoch 2: valid accuracy = 0.7918725974739155  Average loss = 0.4581869826372204  f1_score = 0.7991520932697402
Epoch 3: valid accuracy = 0.7792421746293245  Average loss = 0.4559927089292495  f1_score = 0.8
Epoch 4: valid accuracy = 0.7753981328940143  Average loss = 0.50785831412901  f1_score = 0.7476866132017274
Epoch 5: valid accuracy = 0.7957166392092258  Average loss = 0.44782947960355485  f1_score = 0.7910112359550562
test accuracy = 0.8027522935779816  Average loss = 0.445460513911464  f1_score = 0.7985948477751758
CM [[359  85]
 [ 87 341]]


 Baseline Model

 Epoch 1: valid accuracy = 0.6496430532674354  Average loss = 0.6604076086497698  f1_score = 0.5281065088757396
Epoch 2: valid accuracy = 0.7397034596375618  Average loss = 0.5682536840764552  f1_score = 0.7407002188183808
Epoch 3: valid accuracy = 0.7611202635914333  Average loss = 0.5030679458477458  f1_score = 0.7706905640484977
Epoch 4: valid accuracy = 0.771554091158704  Average loss = 0.4708229167376711  f1_score = 0.7828810020876825
Epoch 5: valid accuracy = 0.7841845140032949  Average loss = 0.4549598881944281  f1_score = 0.7945635128071094
test accuracy = 0.7775229357798165  Average loss = 0.4674034846777266  f1_score = 0.7815315315315315
CM [[331 113]
 [ 81 347]]


 ******************************************************************************************************************************




 The selected best set of hyperparamters
 Gradient clipping value = 1.0
 num_layers = 2
 hidden_size = 150
 dropout = 0
 bidirectional = True 
 batch_size = 10
 vocubulary size = -1
 min_freq = 1
 learning rate = 1e-4
 hidden_size = 150
 optimizer = Adam
 Activation function = ReLU
 pooling type = mean pooling
 Freezing vector representation = True





 seed = 7052020
 LSTM
Epoch 1: valid accuracy = 0.7902251510159253  Average loss = 0.44907253418789533  f1_score = 0.7932900432900432
Epoch 2: valid accuracy = 0.8034047226798462  Average loss = 0.4296200981597757  f1_score = 0.7997762863534676
Epoch 3: valid accuracy = 0.7995606809445359  Average loss = 0.4327748902752751  f1_score = 0.8085998951232303
Epoch 4: valid accuracy = 0.8061504667764964  Average loss = 0.442191241787431  f1_score = 0.8050800662617338
Epoch 5: valid accuracy = 0.8039538714991763  Average loss = 0.4551765984814879  f1_score = 0.7871198568872987
test accuracy = 0.8142201834862385  Average loss = 0.44671537075191736  f1_score = 0.7999999999999999
CM [[386  58]
 [104 324]]

 
 Baseline Model
Epoch 1: valid accuracy = 0.7737506864360242  Average loss = 0.49237295838653067  f1_score = 0.7799145299145299
Epoch 2: valid accuracy = 0.7841845140032949  Average loss = 0.44930278538354756  f1_score = 0.7846575342465755
Epoch 3: valid accuracy = 0.7918725974739155  Average loss = 0.43997239292970775  f1_score = 0.7876750700280113
Epoch 4: valid accuracy = 0.7891268533772653  Average loss = 0.4385859995591836  f1_score = 0.7944325481798715
Epoch 5: valid accuracy = 0.7885777045579352  Average loss = 0.44455261919342104  f1_score = 0.7783534830166954
test accuracy = 0.7763761467889908  Average loss = 0.46079697857864876  f1_score = 0.7642079806529625
CM [[361  83]
 [112 316]]


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

 Baseline Model
Epoch 1: valid accuracy = 0.7638660076880834  Average loss = 0.4956541229141215  f1_score = 0.7769709543568466
Epoch 2: valid accuracy = 0.7896760021965953  Average loss = 0.4482477142022607  f1_score = 0.7919608908202064
Epoch 3: valid accuracy = 0.7797913234486545  Average loss = 0.45300779805157354  f1_score = 0.7600239377618192
Epoch 4: valid accuracy = 0.7808896210873146  Average loss = 0.44910561255703535  f1_score = 0.7948586118251928
Epoch 5: valid accuracy = 0.7973640856672158  Average loss = 0.4387953638867602  f1_score = 0.7935086737548964
test accuracy = 0.7763761467889908  Average loss = 0.45187289572574874  f1_score = 0.769775678866588
CM [[351  93]
 [102 326]]


seed  = 15200
 LSTM
Epoch 1: valid accuracy = 0.7704557935200439  Average loss = 0.46517856704081345  f1_score = 0.7901606425702812
Epoch 2: valid accuracy = 0.7797913234486545  Average loss = 0.445569349239107  f1_score = 0.8007948335817189
Epoch 3: valid accuracy = 0.7924217462932455  Average loss = 0.4312638890384976  f1_score = 0.8079268292682926
Epoch 4: valid accuracy = 0.8034047226798462  Average loss = 0.4818784069738101  f1_score = 0.7956621004566209
Epoch 5: valid accuracy = 0.800109829763866  Average loss = 0.43504565490073843  f1_score = 0.7881257275902213
test accuracy = 0.8222477064220184  Average loss = 0.42294580497863615  f1_score = 0.8130277442702051
CM [[380  64]
 [ 91 337]]

 Baseline Model
Epoch 1: valid accuracy = 0.7655134541460736  Average loss = 0.5048134722670571  f1_score = 0.7404255319148936
Epoch 2: valid accuracy = 0.7874794069192751  Average loss = 0.4522199726821295  f1_score = 0.7792355961209355
Epoch 3: valid accuracy = 0.7841845140032949  Average loss = 0.44067070897811095  f1_score = 0.7932667017359284
Epoch 4: valid accuracy = 0.7874794069192751  Average loss = 0.44593785080264825  f1_score = 0.7951296982530439
Epoch 5: valid accuracy = 0.7688083470620538  Average loss = 0.4686077535966706  f1_score = 0.7380211574362165
test accuracy = 0.7740825688073395  Average loss = 0.4789147869768468  f1_score = 0.7431551499348109
CM [[390  54]
 [143 285]]

seed = 158200
 LSTM
Epoch 1: valid accuracy = 0.7946183415705657  Average loss = 0.4704762917887318  f1_score = 0.7945054945054945
Epoch 2: valid accuracy = 0.7957166392092258  Average loss = 0.435661943313854  f1_score = 0.8070539419087137
Epoch 3: valid accuracy = 0.8050521691378364  Average loss = 0.42317931516900087  f1_score = 0.8008973639932699
Epoch 4: valid accuracy = 0.8028555738605162  Average loss = 0.4385459483728383  f1_score = 0.8154241645244217
Epoch 5: valid accuracy = 0.785831960461285  Average loss = 0.4569052525190382  f1_score = 0.8088235294117647
test accuracy = 0.7775229357798165  Average loss = 0.4949652143669399  f1_score = 0.797071129707113
CM [[297 147]
 [ 47 381]]

 Baseline Model
Epoch 1: valid accuracy = 0.7836353651839648  Average loss = 0.4936221202865976  f1_score = 0.7830396475770925
Epoch 2: valid accuracy = 0.7951674903898956  Average loss = 0.445341606324162  f1_score = 0.7896221094190637
Epoch 3: valid accuracy = 0.7825370675453048  Average loss = 0.44227532720793794  f1_score = 0.7891373801916933
Epoch 4: valid accuracy = 0.7973640856672158  Average loss = 0.437709678829612  f1_score = 0.7932773109243697
Epoch 5: valid accuracy = 0.7962657880285557  Average loss = 0.43453959851459567  f1_score = 0.7949143173023769
test accuracy = 0.7740825688073395  Average loss = 0.44857883419502864  f1_score = 0.7690504103165298
CM [[347  97]
 [100 328]]


seed = 8200
 LSTM
Epoch 1: valid accuracy = 0.7666117517847336  Average loss = 0.4855284222651049  f1_score = 0.7945867568873851
Epoch 2: valid accuracy = 0.7973640856672158  Average loss = 0.43252034778477716  f1_score = 0.8017195056421279
Epoch 3: valid accuracy = 0.7935200439319056  Average loss = 0.4418757756143971  f1_score = 0.8031413612565446
Epoch 4: valid accuracy = 0.8121911037891268  Average loss = 0.41554563745978423  f1_score = 0.8167202572347266
Epoch 5: valid accuracy = 0.785282811641955  Average loss = 0.49376839071402806  f1_score = 0.7617306520414382
test accuracy = 0.801605504587156  Average loss = 0.48891081838783895  f1_score = 0.7818411097099622
CM [[389  55]
 [118 310]]

 Baseline Model
Epoch 1: valid accuracy = 0.7682591982427238  Average loss = 0.496766850108006  f1_score = 0.7835897435897435
Epoch 2: valid accuracy = 0.7803404722679846  Average loss = 0.44604080559207443  f1_score = 0.7920997920997921
Epoch 3: valid accuracy = 0.7891268533772653  Average loss = 0.4376828942061122  f1_score = 0.7955271565495208
Epoch 4: valid accuracy = 0.7924217462932455  Average loss = 0.43746327490754466  f1_score = 0.7987220447284346
Epoch 5: valid accuracy = 0.7902251510159253  Average loss = 0.4362412542918992  f1_score = 0.79831045406547
test accuracy = 0.7763761467889908  Average loss = 0.45056928355585446  f1_score = 0.7845303867403314
CM [[322 122]
 [ 73 355]]

 

The average and deviation of all tracked metrics


LSTM
Accuracy: {'average': 0.80366, 'std_dev': 0.015127141170756649}
Loss:     {'average': 0.45792, 'std_dev': 0.028855113931502682} 
f1_score: {'average': 0.79828, 'std_dev': 0.00994090539136146}}

Baseline Model 
Accuracy: {'average': 0.77548, 'std_dev': 0.0011267652816802465} 
Loss:     {'average': 0.45816, 'std_dev': 0.011179016056880849}
f1_score: {'average': 0.76616, 'std_dev': 0.013336056388603048}}
"""