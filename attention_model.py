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

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded, hidden)

        return outputs, hidden
    
    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
    
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, device, dropout_p=0.1, max_length=50):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        h_dec = hidden[0][-1].unsqueeze(0)
        attn_input = torch.cat((embedded[0], h_dec[0]), 1)
        attn_weights = F.softmax(self.attn(attn_input), dim=1)

        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights