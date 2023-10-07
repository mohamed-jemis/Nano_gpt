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


class EncoderBlock(nn.Module):
    def __init__(self,SelfAttentionBlock: MultiHeadAttention, FeedForwardBlock: FeedForward, dropout: float) -> None:
        super( self).__init__()
        self.SelfAttentionBlock = SelfAttentionBlock
        self.FeedForwardBlock = FeedForwardBlock
        self.ResidualConnection = nn.ModuleList([ResidualConnection(dropout)for i in range(2)])

    def forward(self, x, mask):
        x = self.ResidualConnection[0](x, lambda x: self.SelfAttentionBlock(x, x, x, mask))
        x = self.ResidualConnection[1](x, self.FeedForwardBlock)
        # x here is a tensor here it contatins the output from two parts the first is the multiheadattention part and X  second is the feed forward output and X
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(self).__init__()
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,SelfAttentionBlock: MultiHeadAttention,CrossAttentionBlock:MultiHeadAttention, FeedForwardBlock: FeedForward, dropout: float) -> None:
        super().__init__()
        self.SelfAttentionBlock = SelfAttentionBlock
        self.CrossAttentionBlock = CrossAttentionBlock
        self.FeedForwardBlock = FeedForwardBlock
        self.ResidualConnection = nn.ModuleList([ResidualConnection(dropout)for i in range(3)])

    def forward(self, x,encoder_output,encoder_mask,decoder_mask):
        x = self.ResidualConnection[0](x, lambda x: self.SelfAttentionBlock(x, x, x, decoder_mask))
        x = self.ResidualConnection[1](x, lambda x: self.CrossAttentionBlock(x, encoder_output, encoder_output, encoder_mask))
        x = self.ResidualConnection[2](x, self.FeedForwardBlock)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, *args, **kwargs):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization

    def forward(self, x, encoder_output, encoder_mask,decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask,decoder_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self,d_model: int, vocab_size: int)-> None:
        super().__init__()
        self.projection = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder,src_embd_layer:InputEmbedding,target_embd_layer: InputEmbedding,
                 src_pos_encoding: PositionalEncoding,target_pos_encoding: PositionalEncoding,
                 projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd_layer = src_embd_layer
        self.target_embd_layer = target_embd_layer
        self.src_pos_encoding = src_pos_encoding
        self.target_pos_encoding = target_pos_encoding
        self.projection_layer = projection_layer

    def encode(self,src,mask):
        src = self.src_pos_encoding(src)
        src = self.src_embd_layer(src)
        return self.encoder(src, mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        src = self.target_embd_layer(target)
        target = self.target_pos_encoding(target)
        return self.decoder(target, encoder_output,src_mask,target_mask)
        #         this funcntion needs visualization

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size : int, target_vocab_size:int ,
                      src_seq_len: int,target_seq_len: int,
                      d_model: int = 512,N:int =6,h:int=8,
                      dropout:float = 0.1,d_ff : int= 2048)->Transformer:
    src_embd = InputEmbedding(d_model,src_vocab_size)
    target_embd = InputEmbedding(d_model,target_vocab_size)

    src_pos_encoding = PositionalEncoding(d_model, src_seq_len,dropout)
    target_pos_encoding = PositionalEncoding(d_model, target_seq_len,dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model,h,dropout)
        encoder_feedforward = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention,encoder_feedforward,dropout)
        encoder_blocks.append(encoder_block)


    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model,h ,dropout)
        decoder_feedforward = FeedForward(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention,decoder_cross_attention,decoder_feedforward,dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model,target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embd,target_embd,src_pos_encoding,target_pos_encoding,projection_layer)


#   initialize the parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer



