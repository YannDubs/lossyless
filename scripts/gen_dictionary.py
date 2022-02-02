    """generate dictionary from wikitext-2
    
    """
from doctest import Example
import json
import os
import glob
import torch
import torchtext
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

print(torch.__version__)  # 1.3.1
print(torchtext.__version__)  # 0.5.0

tokenizer = get_tokenizer('basic_english')
train_iter = WikiText2(split='train')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

num = 12000
lists = vocab.lookup_tokens(list(range(num)))[100:]

id2word, word2id = {}, {}
removed_word = {}
i = 0

for index, word in enumerate(lists):
    if word.isalpha():
        id2word[index] = word
        word2id[word] = index
        i += 1
    else:
        removed_word[index] = word
    if i == 10000:
        print('Saved 10000 words!')
        break

def save_json(content, file_name):
    if file_name[-5:] != '.json':
        file_name = file_name + '.json'
    print(file_name)
    with open(file_name, 'w') as file:
        json.dump(content, file)
        
save_json(id2word, 'id2word')
save_json(word2id, 'word2id.json')


        
        

        
        