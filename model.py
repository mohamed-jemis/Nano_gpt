import math

import torch
import torch.nn
from torch import nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        # hyper parameter that you can change
        self.vocab_size = vocab_size
        # da be bsata el size bta3 l kalemat 3ndy
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        # according ot the paper


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pos_embed = torch.zeros(seq_len, d_model)
        numerator = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        # we calculated it in log space for numerical stability
        pos_embed[:, 0::2] = torch.sin(numerator * denominator)
        pos_embed[:, 1::2] = torch.cos(numerator * denominator)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer(
            'pos_emd' , pos_embed
        )


    def forward(self,x):
        x = x + (self.pos_embed[:,:x.shape[1], :]).requires_grad_(False)
        # chatgpt said use detach
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
#         parameter de btkhly l haga learnable

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1,keepdim = True)
        return self.alpha * (x- mean) / (std * self.eps) * self.bias




class FeedForward(nn.Module):
    def __init__(self,d_model:int , d_ff:int , dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0,"d_model not divisble by head count"
        self.dk = d_model / h
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        self.Wo = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e11)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ v), attention_scores


    def forward(self, q, k, v, mask):
        Query = self.Wq(q) # (batch len ,d_model)
        Key = self.Wk(k)
        Value = self.Wv(v)

        #split the embeddings
        # from (b , seq_len ,d_model) -> (b, seq_len ,h,d_k) -> (n,h,seq_len,d_k)

        Query = Query.view(Query.shape[0], Query.shape[1], self.h, self.d_k).transpose(1, 2)
        Key = Key.view(Key.shape[0], Key.shape[1], self.h, self.d_k).transpose(1, 2)
        Value = Value.view(Value.shape[0], Value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(Query, Key, Value, mask, self.dropout)
        # (batch , h ,seq_len,d_k) -> (batch , seq_len,h,d_model) -> (batch,h,seq_len,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.dk)

        return self.Wo(x)


class ResidualConnection(nn.Module):

    def __init__(self,dropout:float)-> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer
                                (self.norm(x)))
    # in the paper its add Norm but here we norm first then add 
