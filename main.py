from typing import Union

from fastapi import FastAPI
import torch
from utils_fast import *
import torch.nn as nn
from torch.nn import functional as F
from fastapi.middleware.cors import CORSMiddleware

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 48 # what is the maximum context length for predictions?
max_iters = 30_000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)




# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s




# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            p = probs.tolist()[0]
            ls = sort_index(p)[:10]
            return p,ls


import __main__
setattr(__main__, "BigramLanguageModel", BigramLanguageModel)
setattr(__main__, "Block", Block)
setattr(__main__, "MultiHeadAttention", MultiHeadAttention)
setattr(__main__, "Head", Head)
setattr(__main__, "FeedFoward", FeedFoward)






# load
import json

with open('./model/token-encode-20000.json', 'r') as file:
    stoi = json.load(file)

with open('./model/token-decode-20000.json', 'r') as file:
    itos = json.load(file)

print(len(stoi), len(itos))

def encode(s):
  ans = []
  s = s.split("/")
  for i in s:
    i = i.split("\n")
    for j in i:
      ans.append(int(stoi[j]))
    ans.append(int(stoi[" "]))
  return ans


# encode = lambda s: [stoi[c+" "] for c in s.split(" ")] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[str(i)] for i in l]) # decoder: take a list of integers, output a string



model_load = torch.load('./model/model-20000',map_location=torch.device('cpu'))



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/predict")
def predict(crypt: str = ""):
    li = encode(crypt)
    a_list = [li]
    a_tensor = torch.Tensor(a_list).to(torch.int64)
    decode(li)


    device = 'cpu' 
    m2 = model_load.to(device)
    p, check = m2.generate(a_tensor, max_new_tokens=2)
    ans = []
    for i in check:
        print(p[i],"%.  ", decode([i]))
        ans.append({"prob": p[i], "hero":decode([i])})

    return {"pred": ans}



@app.get("/test")
def predict():
    li = encode("Shadow Fiend/Monkey King/Enchantress/Hoodwink/Snapfire/Weaver/Sven/Brewmaster/Dragon Knight/Jakiro/Phoenix/Broodmother/Nyx Assassin/Elder Titan/Batrider/Crystal Maiden/Ember Spirit/Ursa/Enigma/Dazzle/Templar Assassin/Dark Seer/Tiny/")
    a_list = [li]
    a_tensor = torch.Tensor(a_list).to(torch.int64)
    decode(li)


    device = 'cpu' 
    m2 = model_load.to(device)
    p, check = m2.generate(a_tensor, max_new_tokens=2)
    ans = []
    for i in check:
        print(p[i],"%.  ", decode([i]))
        ans.append({"prob": p[i], "hero":decode([i])})

    return {"pred": ans}


