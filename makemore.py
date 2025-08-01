words = open('names.txt', 'r').read().splitlines()
# #exemplu
# # for w in words[:1]:
# #   chs = ['<S>'] + list(w) + ['<E>']
# #   for ch1, ch2 in zip(chs, chs[1:]):
# #     print(ch1, ch2)

# b = {}    
# for w in words:
#   chs = ['<S>'] + list(w) + ['<E>']
#   for ch1, ch2 in zip(chs, chs[1:]):
#     bigram = (ch1, ch2)
#     b[bigram] = b.get(bigram, 0) + 1
#print(sorted(b.items(), key=lambda kv:-kv[1]))

import torch

# N = torch.zeros((27, 27), dtype=torch.int32) # 26 litere alfabet englez + '.'

chars = sorted(list(set(''.join(words))))
# #print(chars)
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
# #print(stoi)
itos = {i:s for s,i in stoi.items()}
# #print(itos)
# for w in words:
#   chs = ['.'] + list(w) + ['.']
#   for ch1, ch2 in zip(chs, chs[1:]):
#     ix1 = stoi[ch1]
#     ix2 = stoi[ch2]
#     N[ix1, ix2] += 1

# import matplotlib.pyplot as plt

# plt.figure(figsize=(20,20))
# plt.imshow(N, cmap='Blues', aspect='equal', interpolation='none', alpha=0.6)
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.annotate(f"{chstr}\n{N[i,j].item()}", (j, i), ha = "center", va = "center", color = 'black', fontsize = 6)
# plt.axis('off');
# plt.show()

# print(N[0])

# converteste la probabilitati
# p = N[0].float()
# p = p / p.sum()
# # print(p)

# g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# # print(itos[ix])

# g = torch.Generator().manual_seed(2147483647)
# p = torch.rand(3, generator=g)
# p = p / p.sum()
# # print(p)

# # print(torch.multinomial(p, num_samples=100, replacement=True, generator=g))

# g = torch.Generator().manual_seed(2147483647)

# P = (N+1).float()
# P /= P.sum(1, keepdims=True)

# for i in range(20):
  
#   out = []
#   ix = 0
#   while True:
#     p = P[ix]
#     ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#     out.append(itos[ix])
#     if ix == 0:
#       break
#   print(''.join(out))

#   #EVALUAREA EFICIENTEI MODELULUI
# log_likelihood = 0.0
# n = 0
# for w in words:
#   chs = ['.'] + list(w) + ['.']
#   for ch1, ch2 in zip(chs, chs[1:]):
#     ix1 = stoi[ch1]
#     ix2 = stoi[ch2]
#     prob = P[ix1, ix2]
#     n += 1
#     logprob = torch.log(prob)
#     log_likelihood += logprob
#     # print(f'{ch1}{ch2}:{prob:.4f}{logprob:.4f}')
# # print(f'{log_likelihood}')    
# nll = -log_likelihood
# print(f'{nll=}') 
# print(f'{nll/n}')   
#   negative log likelihood - folosit ca functie loss- cu cat e mai aproape de 0 cu atat modelul e mai eficient, 
#                 cu cat e mai mare, cu atat eficienta modeluilui e mai rea
 
# MAXIMUM LOG LIKELIHOOD ESTIMATION - produsul tuturor probabilitatilor individuale - 
# 'probabilitatea intregului set de date' - exprima calitatea modelului

# OBIECTIV: maximizarea log likelihood (probabilitatii logaritmice) datelor în raport cu parametrii modelului (modelare statistică)
# echivalentă cu maximizarea log-likelihood (deoarece logaritmul este funtie monotona)
# echivalentă cu minimizarea log-likelihood negative
# echivalentă cu minimizarea mediei log-likelihood negative

#Implementation using NN
#Create the training set of all the bigrams
xs, ys = [], []

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    # print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
    
xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F
# xenc = F.one_hot(xs, num_classes=27).float()
# print(xenc)

# W = torch.randn((27, 27))
# reshaped_tensor = W.view(-1)
# xenc @ W
# logits = xenc @ W # log-counts
# counts = logits.exp() # equivalent N
# probs = counts / counts.sum(1, keepdims=True)
# reshaped_tensor = probs.view(-1)


# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# btw: the last 2 lines here are together called a 'softmax'
print(probs.shape)
