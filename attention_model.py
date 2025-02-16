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
    
    def initHidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
    
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
        self.dropout = nn.Dropout(self.dropout_p)

        self.attn_encoder = nn.Linear(hidden_size, hidden_size)
        self.attn_decoder = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, encoder_outputs, encoder_mask=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        decoder_hidden = hidden[0][-1]
        decoder_hidden = decoder_hidden.unsqueeze(1)

        encoder_outputs = encoder_outputs.transpose(0, 1)

        energy = torch.tanh(self.attn_encoder(encoder_outputs) + self.attn_decoder(decoder_hidden))
        attn_scores = self.attn_v(energy).squeeze(2)

        if encoder_mask is not None:
            attn_scores = attn_scores.masked_fill(~encoder_mask, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.transpose(0, 1)

        output = torch.cat((embedded, context), dim=2)
        output = self.attn_combine(output)
        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)
        output = self.out(output.squeeze(0))
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights