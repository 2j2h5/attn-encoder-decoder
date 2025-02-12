import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter

from data import *
from model import Encoder, Decoder
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
max_target_length = 40
hidden_size = 256
num_layers = 1
learning_rate = 0.0005
teacher_forcing_ratio = 0.7
num_iters = 1000
print_every = 10
num_samples = 10
clip_value = 5.0

SOS_token = tgt_W2I["<SOS>"]
EOS_token = tgt_W2I["<EOS>"]
UNK_token = tgt_W2I["<UNK>"]

loss_list = []
writer = SummaryWriter(log_dir="./runs/gradient_visualization/encdec")

encoder = Encoder(src_WrdCnt, hidden_size, num_layers=num_layers, device=device).to(device)
decoder = Decoder(hidden_size, tgt_WrdCnt, num_layers=num_layers, device=device).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss() 

def train(input_tensor, target_tensor, use_teacher_forcing_flag=True):
    h, c = encoder.initHidden()
    encoder_hidden = (h, c)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    for ei in range(input_length):
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio and use_teacher_forcing_flag else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_value)
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(input_tensor):
    with torch.no_grad():
        h, c = encoder.initHidden()
        encoder_hidden = (h, c)
        input_length = input_tensor.size(0)

        for ei in range(input_length):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            token = topi.item()

            if token == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(tgt_I2W[token])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluate_bleu(pairs, num_samples=100):
    references = []
    hypotheses = []

    smoothing_fn = SmoothingFunction().method4

    for _ in range(num_samples):
        pair = random.choice(pairs)
        input_tensor = tensorFromSentence(pair[0], src_W2I, device=device)
        output_words = evaluate(input_tensor)

        if output_words and output_words[-1] == "<EOS>":
            output_words = output_words[:-1]

        references.append([pair[1].split()])
        hypotheses.append(output_words)

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
    return bleu_score

if __name__ == "__main__":
    print("Training...")
    for iter in range(num_iters):
        pair = random.choice(train_pairs)
        input_tensor = tensorFromSentence(pair[0], src_W2I, device=device)
        target_tensor = tensorFromSentence(pair[1], tgt_W2I, device=device)

        loss = train(input_tensor, target_tensor, use_teacher_forcing_flag=True)    
        loss_list.append(loss)

        if iter % print_every == 0:
            for name, param in encoder.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"encoder_gradients/{name}", param.grad.cpu().data.numpy(), iter)
                    writer.add_scalar(f"encoder_gradients_norm/{name}", param.grad.data.norm(2).item(), iter)

            for name, param in decoder.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"decoder_gradients/{name}", param.grad.cpu().data.numpy(), iter)
                    writer.add_scalar(f"decoder_gradients_norm/{name}", param.grad.data.norm(2).item(), iter)
            print(f"Iteration {iter}, Loss: {loss:.4f}")
        
    writer.close()

    for _ in range(num_samples):
        sample_pair = random.choice(valid_pairs)
        input_tensor = tensorFromSentence(sample_pair[0], src_W2I, device=device)
        output_words = evaluate(input_tensor)
        print("============================")
        print(f"Input(en): {sample_pair[0]}")
        print("Output(de):", " ".join(output_words))
        print(f"Target(de): {sample_pair[1]}")

    bleu = evaluate_bleu(valid_pairs, num_samples=100)
    print(f"BLEU score (corpus): {bleu * 100:.2f}")

    x = torch.arange(num_iters)
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_list, label='Loss', linestyle='-', color='blue')
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    plt.show()