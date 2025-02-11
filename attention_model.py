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
        self.outputs = None

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        self.outputs.append(output)

        return output, hidden
    
    def initHidden(self):
        self.outputs = []
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device, dropout_p=0.1, max_length=80):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        decoder_hidden_top = hidden[0][-1]

        attn_input = torch.cat((embedded[0], decoder_hidden_top), 1)
        attn_weights = F.softmax(self.attn(attn_input), dim=1)

        seq_len = encoder_outputs.size(0)
        if seq_len < self.max_length:
            padding = torch.zeros(self.max_length - seq_len, 1, self.hidden_size, device=self.device)
            encoder_outputs_padded = torch.cat((encoder_outputs, padding), dim=0)
        else:
            encoder_outputs_padded = encoder_outputs[:self.max_length]

        encoder_outputs_padded = encoder_outputs_padded.squeeze(1)

        attn_weights_ = attn_weights.unsqueeze(0)
        encoder_outputs_padded_ = encoder_outputs_padded.unsqueeze(0)
        attn_applied = torch.bmm(attn_weights_, encoder_outputs_padded_)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden, attn_weights.squeeze(0)
    
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)