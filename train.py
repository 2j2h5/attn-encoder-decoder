import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import *
from model import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 80
SOS_token = 0
EOS_token = 1
UNK_token = 2
n_iters = 20
num_layers = 4

loss_list = []

with open("dataset.pkl", "rb") as f:
    loaded_data = pickle.load(f)

train_pairs = loaded_data["train_pairs"]
valid_pairs = loaded_data["valid_pairs"]
test_pairs = loaded_data["test_pairs"]
fr_W2I = loaded_data["fr_W2I"]
fr_I2W = loaded_data["fr_I2W"]
fr_W2C = loaded_data["fr_W2C"]
fr_WrdCnt = loaded_data["fr_WrdCnt"]
en_W2I = loaded_data["en_W2I"]
en_I2W = loaded_data["en_I2W"]
en_W2C = loaded_data["en_W2C"]
en_WrdCnt = loaded_data["en_WrdCnt"]

def indexesFromSentence(sentence,word2index):
    indice = []
    for word in sentence.split():
        if word in word2index:
            indice.append(word2index[word])
        else:
            indice.append(word2index["UNK"])
    return indice

def tensorFromSentence(sentence,word2index):
    indexes = indexesFromSentence(sentence,word2index)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0],en_W2I)
    target_tensor = tensorFromSentence(pair[1],fr_W2I)
    return (input_tensor, target_tensor)

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_length):
    h, c = encoder.initHidden()
    encoder_hidden = (h, c)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    for ei in range(input_tensor.size(0)):
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.size(0)

def train(encoder, decoder, n_iters=n_iters, learning_rate=0.01):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() 

    for iter in range(n_iters):
        pair = random.choice(train_pairs)
        input_tensor = tensorFromSentence(pair[0], fr_W2I)
        target_tensor = tensorFromSentence(pair[1], en_W2I)

        loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_list.append(loss)

        print(f"Iteration {iter}, Loss: {loss:.4f}")

hidden_size = 256
print(fr_WrdCnt, en_WrdCnt)
encoder = Encoder(fr_WrdCnt, hidden_size, num_layers=num_layers, device=device).to(device)
decoder = Decoder(hidden_size, en_WrdCnt, num_layers=num_layers, device=device).to(device)

train(encoder, decoder, n_iters=10000)

x = torch.arange(n_iters)
plt.figure(figsize=(10, 6))

plt.plot(x, loss_list, label='Loss', linestyle='-', color='blue')

plt.xlabel("Iters")
plt.ylabel("Loss")
plt.title("Loss Curves")

plt.grid(True)

plt.show()