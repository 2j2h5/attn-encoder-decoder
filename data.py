import re
import pickle
from datasets import load_dataset

def load_data(percent=0.1):
    dataset = load_dataset("wmt14", "fr-en", split=f"train[:{int(percent*100)}%]")

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))

    train_dataset = dataset.select(range(0, train_size))
    valid_dataset = dataset.select(range(train_size, train_size + valid_size))
    test_dataset = dataset.select(range(train_size + valid_size, len(dataset)))

    return train_dataset, valid_dataset, test_dataset

def clean_text(text):
    text = re.sub(r'(\d+)', '', text)
    text = re.sub(r'[,.!?;:()\"\']‘’“”', '', text)
    text = text.lower().strip()

    return text

def get_sentences(dataset, src_lang, tgt_lang):
    src_sentences = [clean_text(ex['translation'][src_lang]) for ex in dataset]
    tgt_sentences = [clean_text(ex['translation'][tgt_lang]) for ex in dataset]

    return src_sentences, tgt_sentences

def get_sentence_pairs(src_sentences, tgt_sentences):
    pairs = []
    for src, tgt in zip(src_sentences, tgt_sentences):
        reversed_src = ' '.join(reversed(src.split()))
        pairs.append([reversed_src, tgt])
    
    return pairs

def vectorize(dataset):
    word2index = {"<SOS>": 0, "<EOS>": 1, "<UNK>": 2}
    index2word = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>"}
    word2count = {}
    cnt = 3

    for sentence in dataset:
        for word in sentence.split():
            if word not in word2index:
                word2index[word] = cnt
                index2word[cnt] = word
                word2count[word] = 1
                cnt += 1
            else:
                word2count[word] += 1

    print(f"Number of words: {cnt}")
    return word2index, index2word, word2count, cnt
    
if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_data(percent=0.01)

    print("Extracting sentences...")
    train_src, train_tgt = get_sentences(train_dataset, 'en', 'fr')
    valid_src, valid_tgt = get_sentences(valid_dataset, 'en', 'fr')
    test_src, test_tgt = get_sentences(test_dataset, 'en', 'fr')

    print("Reversing input sentences...")
    train_pairs = get_sentence_pairs(train_src, train_tgt)
    valid_pairs = get_sentence_pairs(valid_src, valid_tgt)
    test_pairs = get_sentence_pairs(test_src, test_tgt)

    print("Vectorizing...")
    en_W2I, en_I2W, en_W2C, en_WrdCnt = vectorize(train_src + valid_src + test_src)
    fr_W2I, fr_I2W, fr_W2C, fr_WrdCnt = vectorize(train_tgt + valid_tgt + test_tgt)

    data_to_save = {
        "train_pairs": train_pairs,
        "valid_pairs": valid_pairs,
        "test_pairs": test_pairs,
        "fr_W2I": fr_W2I,
        "fr_I2W": fr_I2W,
        "fr_W2C": fr_W2C,
        "fr_WrdCnt": fr_WrdCnt,
        "en_W2I": en_W2I,
        "en_I2W": en_I2W,
        "en_W2C": en_W2C,
        "en_WrdCnt": en_WrdCnt
    }

    with open("dataset.pkl", "wb") as f:
        pickle.dump(data_to_save, f)