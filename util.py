import torch

SOS_token = 0
EOS_token = 1
UNK_token = 2

def indexesFromSentence(sentence, W2I):
    indice = []
    for word in sentence.split():
        if word in W2I:
            indice.append(W2I[word])
        else:
            indice.append(W2I["<UNK>"])
    return indice

def tensorFromSentence(sentence, W2I, device):
    indexes = indexesFromSentence(sentence, W2I)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, src_W2I, tgt_W2I, device):
    input_tensor = tensorFromSentence(pair[0], src_W2I, device)
    target_tensor = tensorFromSentence(pair[1], tgt_W2I, device)
    return (input_tensor, target_tensor)