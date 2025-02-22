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

print(f"Vocab Size({src_lang}): {src_WrdCnt}")
print(f"Vocab Size({tgt_lang}): {tgt_WrdCnt}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_target_length = 40
hidden_size = 1000
num_layers = 1
learning_rate = 0.0005
teacher_forcing_ratio = 0.7
num_iters = 10000
print_every = 10
num_samples = 10
clip_value = 5.0
batch_size = 80

SOS_token = tgt_W2I["<SOS>"]
EOS_token = tgt_W2I["<EOS>"]
UNK_token = tgt_W2I["<UNK>"]
PAD_token = tgt_W2I["<PAD>"]

loss_list = []
writer = SummaryWriter(log_dir="./runs/gradient_visualization/encdec")

encoder = Encoder(src_WrdCnt, hidden_size, num_layers=num_layers, device=device).to(device)
decoder = Decoder(hidden_size, tgt_WrdCnt, num_layers=num_layers, device=device).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def prepare_batch(pairs, batch_size):
    batch_pairs = random.sample(pairs, batch_size)
    input_sequences = []
    target_sequences = []

    for pair in batch_pairs:
        input_seq = indexesFromSentence(pair[0], src_W2I)
        input_seq.append(EOS_token)
        target_seq = indexesFromSentence(pair[1], tgt_W2I)
        target_seq.append(EOS_token)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    max_input_len = max(len(seq) for seq in input_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)

    padded_inputs = [seq + [PAD_token]*(max_input_len - len(seq)) for seq in input_sequences]
    padded_targets = [seq + [PAD_token]*(max_target_len - len(seq)) for seq in target_sequences]

    input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device).transpose(0,1)
    target_tensor = torch.tensor(padded_targets, dtype=torch.long, device=device).transpose(0,1)
    input_lengths =[len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]

    return input_tensor, input_lengths, target_tensor, target_lengths

def train(input_tensor, input_lengths, target_tensor, target_lengths, use_teacher_forcing_flag=True):
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    _, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token] * batch_size], dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden

    loss = 0
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio and use_teacher_forcing_flag else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di].unsqueeze(0)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.transpose(0, 1)
            loss += criterion(decoder_output, target_tensor[di])
            if (decoder_input == EOS_token).all():
                break

    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_value)
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    total_token = sum(target_lengths)
    return loss.item() / total_token

def evaluate(input_tensor, input_lengths):
    with torch.no_grad():
        current_batch_size = input_tensor.size(1)
        encoder_hidden = encoder.initHidden(current_batch_size)
        _, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * current_batch_size], dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden

        decoded_sentences = [[] for _ in range(current_batch_size)]
        finished = [False] * current_batch_size

        for di in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.transpose(0, 1)

            for i in range(current_batch_size):
                if not finished[i]:
                    token = decoder_input[0, i].item()
                    if token == EOS_token:
                        decoded_sentences[i].append("<EOS>")
                        finished[i] = True
                    else:
                        decoded_sentences[i].append(tgt_I2W.get(token, "<UNK>"))
            if all(finished):
                break

        return decoded_sentences

def evaluate_bleu(pairs, num_samples=100):
    references = []
    hypotheses = []

    smoothing_fn = SmoothingFunction().method4

    for _ in range(num_samples):
        pair = random.choice(pairs)
        input_tensor, input_lengths, _, _ = prepare_batch([pair], 1)
        output_words = evaluate(input_tensor, input_lengths)[0]

        if output_words and output_words[-1] == "<EOS>":
            output_words = output_words[:-1]

        references.append([pair[1].split()])
        hypotheses.append(output_words)

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
    return bleu_score

if __name__ == "__main__":
    print("Training...")
    for iter in range(num_iters):
        input_tensor, input_lengths, target_tensor, target_lengths = prepare_batch(train_pairs, batch_size)

        loss = train(input_tensor, input_lengths, target_tensor, target_lengths, use_teacher_forcing_flag=True)    
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

    sample_batch = prepare_batch(valid_pairs, batch_size)
    decoded_sentences = evaluate(sample_batch[0], sample_batch[1])
    print("Sample Evaluation on Valid Batch:")
    for i, sent in enumerate(decoded_sentences[:5]):
        print(f"Input(en): {valid_pairs[i][0]}")
        print("Output(de):", " ".join(sent))
        print(f"Target(de): {valid_pairs[i][1]}")
        print("-----")

    print("Evaluating BLEU on valid set...")
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