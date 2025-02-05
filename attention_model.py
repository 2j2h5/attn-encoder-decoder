import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device =device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, x, h):
        embedded = self.embedding(x).view(1, 1, -1)
        y = embedded
        y, h = self.lstm(y, h)

        return y, h
    
    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        y = self.embedding(x).view(1, 1, -1)
        y = F.relu(y)
        y, h = self.lstm(y, h)
        y = self.softmax(self.out(y[0]))

        return y, h
    
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)