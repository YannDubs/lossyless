import json
import torch
import clip

def get_words_list(file_name):
    with open(file_name, 'r') as file:
        word2id = json.load(file)    
    return list(word2id.keys())

words = get_words_list('word2id.json')

embeddings = []
for i in range(0, len(words), 64):
    w = words[i:i + 64]
    embeddings.append(clip.tokenize(w))

output = torch.cat(embeddings)
torch.save(output, 'embeddings.pt')
print('saved! tensor shape: ', output.shape)