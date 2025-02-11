import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import *
from attention_model import Encoder, AttnDecoder
from util import *

pkl_file = "multi30k-en-de.pkl"
src_lang = "en"
tgt_lang = "de"

with open(pkl_file, "rb") as f:
    loaded_data = pickle.load(f)

train_pairs = loaded_data["train_pairs"]
valid_pairs = loaded_data["valid_pairs"]
test_pairs = loaded_data["test_pairs"]
src_W2I = loaded_data[f"{src_lang}_W2I"]
src_I2W = loaded_data[f"{src_lang}_I2W"]
src_W2C = loaded_data[f"{src_lang}_W2C"]
src_WrdCnt = loaded_data[f"{src_lang}_WrdCnt"]
tgt_W2I = loaded_data[f"{tgt_lang}_W2I"]
tgt_I2W = loaded_data[f"{tgt_lang}_I2W"]
tgt_W2C = loaded_data[f"{tgt_lang}_W2C"]
tgt_WrdCnt = loaded_data[f"{tgt_lang}_WrdCnt"]

print(src_WrdCnt, tgt_WrdCnt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 40
hidden_size = 256
num_layers = 4
learning_rate = 0.001
teacher_forcing_ratio = 0.5
num_iters = 300
print_every = 10

SOS_token = tgt_W2I["<SOS>"]
EOS_token = tgt_W2I["<EOS>"]
UNK_token = tgt_W2I["<UNK>"]

loss_list = []

encoder = Encoder(src_WrdCnt, hidden_size, num_layers=num_layers, device=device).to(device)
attn_decoder = AttnDecoder(hidden_size, tgt_WrdCnt, num_layers=num_layers, device=device, max_length=max_length).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss() 

def train(input_tensor, target_tensor, use_teacher_forcing_flag=True):
    h, c = encoder.initHidden()
    encoder_hidden = (h, c)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    if input_length < max_length:
        padding = torch.zeros(max_length - input_length, 1, encoder.hidden_size, device=device)
        encoder_outputs = torch.cat((encoder_outputs, padding), dim=0)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    loss = 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio and use_teacher_forcing_flag else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di].unsqueeze(0)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach().view(1, -1)
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(input_tensor):
    with torch.no_grad():
        h, c = encoder.initHidden()
        encoder_hidden = (h, c)
        input_length = input_tensor.size(0)

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        if input_length < max_length:
            padding = torch.zeros(max_length - input_length, 1, encoder.hidden_size, device=device)
            encoder_outputs = torch.cat((encoder_outputs, padding), dim=0)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden, attn_weights = attn_decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            token = topi.item()

            if token == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(tgt_I2W[token])

            decoder_input = topi.detach().view(1, -1)

        return decoded_words

if __name__ == "__main__":
    print("Training...")
    for iter in range(num_iters):
        pair = random.choice(train_pairs)
        input_tensor = tensorFromSentence(pair[0], src_W2I, device=device)
        target_tensor = tensorFromSentence(pair[1], tgt_W2I, device=device)

        loss = train(input_tensor, target_tensor, use_teacher_forcing_flag=True)    
        loss_list.append(loss)

        if iter % print_every == 0:
            print(f"Iteration {iter}, Loss: {loss:.4f}")

    sample_pair = random.choice(valid_pairs)
    input_tensor = tensorFromSentence(sample_pair[0], src_W2I, device=device)
    output_words = evaluate(input_tensor)
    print(f"Input(en): {sample_pair[0]}")
    print("Output(de):", " ".join(output_words))
    print(f"Target(de): {sample_pair[1]}")

    x = torch.arange(num_iters)
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_list, label='Loss', linestyle='-', color='blue')
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    plt.show()