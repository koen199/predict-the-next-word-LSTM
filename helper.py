import numpy as np 
from nptyping import Array
import torch
import torch.nn as nn
import torch.optim as optim


def read_file()->str:
    with open('data/anna.txt') as file:
        content = file.read()
    
    return content

def get_data_as_batch(batch_size:int, seq_len:int, data:str):
    data = list(data)
    nbr_sequences_per_batch = int(len(data)/(batch_size*seq_len)) #amount of sequences per batch
    #trim data so it fits
    data = data[0:nbr_sequences_per_batch*seq_len*batch_size]
    np_array = np.asarray(data) 
    #reshape the array so the first dimension is the nbr of batches
    np_array = np.reshape(a=np_array, newshape=(batch_size, -1))

    #The input to the LSTM is 3 dimensional tensor:
    #   -First dimension: batch number
    #   -Second dimension: which charachter in the sequence of length seq_len
    #   -Third dimension: one-hot encoded representation of that charachter

    #The target tensor (desired result) is an indentically shaped tensor but of the next character
    for i in range(0,seq_len*nbr_sequences_per_batch, seq_len):
        input_data = np_array[:, i:i+seq_len]
        if i+seq_len+1 < seq_len*nbr_sequences_per_batch:            
            target = np_array[:, i+1:i+seq_len+1]
        else:
            #Edge case: when you have reached the last charachter, there is no next charachter.... Therefore duplicate the last charachter of the input
            target = np.copy(a=input_data)
            target[:, 0:seq_len-1] = input_data[:,1:seq_len]
        yield input_data, target

def convert_to_one_hot_encoded(np_array, char2int:dict):
    batch_size, seq_len = np_array.shape
    input_size = len(char2int)
    one_hot_array = np.zeros((batch_size, seq_len, input_size), dtype=np.uint8)

    it = np.nditer(np_array, flags=['multi_index'])
    for char in it:
        batch_nbr, sequence_nbr = it.multi_index
        char = np_array[batch_nbr, sequence_nbr]
        one_hot_encoded = one_hot_encode(char2int, char)
        one_hot_array[batch_nbr, sequence_nbr] = one_hot_encoded

    return one_hot_array

def one_hot_encode(char2int:dict, char):
    one_hot_vector = np.zeros((1, len(char2int)), dtype=np.uint8)
    index = char2int[char]
    one_hot_vector[0,index] = 1
    return one_hot_vector


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=hidden_size , out_features=input_size)

    def forward(self, input_, hidden_state):
        output: torch.Tensor
        output, hidden_state = self.lstm(input_, hidden_state)
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden_state

def init_hidden(num_layers, batch_size, hidden_size):             
        h_0 = torch.zeros(num_layers, batch_size, hidden_size)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size)
        hidden_state = (h_0, c_0)
        return hidden_state

def check_data(input_data:torch.Tensor, target:torch.Tensor, char2int):
    #Handy during debugging to see convert input and target tensor back to original char format to
    #see if everything is going well....
    int2char = {value: key for key, value in char2int.items()}
    input_data, target = input_data.numpy(), target.numpy()
    input_data = np.nanargmax(input_data, axis=2)
    input_data = np.reshape(input_data, newshape=(-1))

    input_char = [int2char[int(i)] for i in np.nditer(input_data)]
    target_char = [int2char[int(i)] for i in np.nditer(target)]

def save_model(filename:str, model:LSTM, char2int:dict, use_gpu:bool):
    model = model.cpu()
    checkpoint = {
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'char2int': char2int, 
        'state_dict': model.state_dict()
    }
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
    if use_gpu:
        model = model.cuda()



def train(model:LSTM, char2int:dict, train_data:str, valid_data:str, epochs=5, batch_size=2, seq_len=256, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hidden_state = init_hidden(model.num_layers, batch_size, model.hidden_size)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using gpu for training...")
        model = model.cuda()
    
    model.train()
    counter = 0
    for e in range(epochs):
        min_validation_loss = np.Inf
        for input_data, target in get_data_as_batch(batch_size=batch_size, seq_len=seq_len, data=train_data):
            #Do the below to stop backpropagation trough the entire train data
            hidden_state = tuple([each.data for each in hidden_state])

            #Prepare input to network
            input_data = convert_to_one_hot_encoded(input_data, char2int)
            target = convert_to_one_hot_encoded(target, char2int)
            target = np.nanargmax(target, axis=2)

            #set accumelated gradient to zero
            model.zero_grad()

            #convert numpy arrays to tensors
            input_data, target = torch.Tensor(input_data).type(torch.float32), torch.Tensor(target).type(torch.long)
            #move tensors to the gpu if possible
            if use_gpu:
                h_0, c_0 = hidden_state
                input_data, h_0, c_0, target = input_data.cuda(), h_0.cuda(), c_0.cuda(), target.cuda()
                hidden_state = (h_0, c_0)

            #Forward pass
            output, hidden_state = model.forward(input_data, hidden_state)

            #First dimension is converted to batch_size*seq_len
            target = target.view(-1)

            #calculate loss and backpropagate
            #check_data(input_data, target, char2int)
            loss = criterion(output, target)
            loss.backward()

            #Clip the gradient to avoid the exploding gradient problem
            clip=5
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            
            #increment the step counter
            counter += 1 

            if counter % 500 == 0:
                valid_losses = []
                hidden_state_valid = init_hidden(model.num_layers, batch_size, model.hidden_size)
                model.eval()
                for input_data, target in get_data_as_batch(batch_size=batch_size, seq_len=seq_len, data=valid_data):
                    with torch.no_grad():  
                        #Do the below to stop backpropagation trough the entire train data
                        hidden_state_valid = tuple([each.data for each in hidden_state_valid])

                        #Prepare input to network
                        input_data = convert_to_one_hot_encoded(input_data, char2int)
                        target = convert_to_one_hot_encoded(target, char2int)
                        target = np.nanargmax(target, axis=2)   

                        #convert numpy arrays to tensors
                        input_data, target = torch.Tensor(input_data).type(torch.float32), torch.Tensor(target).type(torch.long)

                        #move tensors to the gpu if possible
                        if use_gpu:
                            h_0, c_0 = hidden_state_valid
                            input_data, h_0, c_0, target = input_data.cuda(), h_0.cuda(), c_0.cuda(), target.cuda()
                            hidden_state_valid = (h_0, c_0)
                        
                        output, hidden_state_valid = model.forward(input_data, hidden_state_valid)
                        target = target.view(-1)
                        valid_loss = criterion(output, target)
                        valid_losses.append(valid_loss.item())


                mean_valid_loss = np.mean(valid_losses)
                print("Epoch: {}/{}...".format(e+1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.4f}...".format(loss.item()),
                "Val Loss: {:.4f}".format(mean_valid_loss))

                if mean_valid_loss < min_validation_loss:
                    save_model('model.pth',model, char2int, use_gpu)
                    print("Lowest validation loss->Saving model!")
                model.train()


            
if __name__ == "__main__":
    data = read_file()
    #split data in train data and validation data
    train_data = data[0:int(0.8*len(data))]
    valid_data = data[int(0.8*len(data)):]

    #Find all unique charachter used in the data
    char_set = set(data)
    #Map each charachter to a unique integer
    char2int = {char: i for i, char in enumerate(char_set)}


    #initialize the model
    input_size = len(char_set)
    model = LSTM(input_size=input_size, hidden_size=256, num_layers=2, dropout=0.2)
    #load model from checkpoint
    #model.load_state_dict(torch.load('model.pth'))

    #train the model
    train(model, char2int, train_data=train_data, valid_data=valid_data, epochs=15, batch_size=3, seq_len=256, lr=0.001)
