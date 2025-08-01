import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

print(words[:8])

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
print(itos)

block_size = 3
X, Y = [], []
for w in words[:5]:
    print (w)
    context = [0]*block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.rand((27, 2 ))

