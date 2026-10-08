import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass # a quick way of initiating a class with attributes
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None # for group KV
    vocab_size: int = -1 # set when loading tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # for KV Cache
    max_batch_size: int = 4
    max_seq_len: int = 1024

    device: Optional[str] = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "the embedding dim must be even for RoPE to work"
    # shape of of all angle values, to differentiate the features in the dim, still m-n overall for the embedding
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # shape: (head_dim / 2)

    # compute for all RoPE angle vectors given any n from 0 to seq_len-1
    m = torch.arange(seq_len, device = device)
    freqs = torch.outer(m, theta).float() # shape: (max_seq_len, head_dim / 2)

    # compute R*exp(i * m * theta), torch.ones_like to indicate R=1
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # shape: (max_seq_len, head_dim/2)

    # freqs_complex_ij = R*exp(i * freqs_ij) = cos(freqs_ij) + i sin(freqs_ij); i is m and j is theta
    # so if want to get the two outer theta vectors, we can first get freqs_complex[m] and do some equivalent computations using these polar stuff
    # converting x to x1+ix2, x3+ix4 style and multiply element wise with freqs_complex then flatten achieves the computation efficient form result listed in paper
    return freqs_complex 

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # onverting x to x1+ix2, x3+ix4 style
    # x shape: (batch, seq_len, num_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1 ,2)) # (batch, seq_len, num_heads, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, head_dim // 2)

    x_rotated = x_complex * freqs_complex # * is the element wise product
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)

    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return x.unsqueeze(-2).expand( # using None just inserts a brandnew dimension
             batch_size, seq_len, n_kv_heads, n_rep, head_dim
             ).reshape(
                batch_size, seq_len, n_kv_heads * n_rep, head_dim
             )

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor): # x shape: (batch, seq_len, dim = num_heads * head_dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) # rsqrt = 1 / sqrt
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # number of kv groups
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # indicate number of qs per group = how many times the keys and values should be repeated to match the head of the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device, dtype=torch.float16) 
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device, dtype=torch.float16) 
    
    # seq len 1 assume just decoding, training needs to be like normal full
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (batch, 1, dim)

        xq = self.wq(x) # (batch, 1, self.n_heads_q * head_dim)
        xk = self.wk(x) # (batch, 1, self.n_kv_heads * head_dim)
        xv = self.wv(x) # (batch, 1, self.n_kv_heads * head_dim)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # no rotary on v
        xq = apply_rotary_embeddings(xq, freqs_complex, device = x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device = x.device)

        # kv cache handling
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # compute
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2) # (batch, 1, self.n_heads_q, head_dim) -> (batch, self.n_heads_q, 1, head_dim)
        keys = keys.transpose(1, 2) # (batch, self.n_heads_q, seq_len_kv, head_dim)
        values = values.transpose(1, 2) # (batch, self.n_heads_q, seq_len_kv, head_dim), seq_len_kv is number of entries in kv cache

        # (batch, self.n_heads_q, 1, head_dim) @ (batch, self.n_heads_q, head_dim, seq_len_kv) -> (batch, self.n_heads_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3) / math.sqrt(self.head_dim))
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        # (batch, self.n_heads_q, 1, seq_len_kv) @ (batch, self.n_heads_q, seq_len_kv, head_dim) -> (batch, self.n_heads_q, 1, head_dim)
        output = torch.matmul(scores, values)

        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)) # (batch, 1, self.n_heads_q * head_dim = hidden_dim)
        return self.wo(output)

class FeedForward(nn.Module): # feed forward with SwiGLU
    
    def __init__(self, args:ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # round to nearest multiple number of params
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False) # this FFN uses three Ws

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads 

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps) # need to separate norms because of the weight
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # x shape: (batch, seq_len, seq_dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex) # apply RoPE for q,k within attention
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps = args.norm_eps) # self defined class
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        # for RoPE compute compute, RoPE only apply on Q,K, which is intuitive indeed
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time, we use KV Cache"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # get pairs (m, theta) for positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len] 

        # consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h) # one last norm in the end before classification head
        output = self.output(h).float()
        return output