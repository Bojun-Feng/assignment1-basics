import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import reduce

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = nn.Parameter(
            torch.zeros(out_features, in_features, device = device, dtype = dtype)
        )

        std = (2.0 / (out_features + in_features)) ** 0.5
        init.trunc_normal_(self.w, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def use_weight(self, w):
        with torch.no_grad():
            self.w.copy_(w)

    def forward(self, x):
        return torch.matmul(x, self.w.t())

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.w = nn.Parameter(
            torch.zeros(num_embeddings, embedding_dim, device = device, dtype = dtype)
        )

        std = 1
        init.trunc_normal_(self.w, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def use_weight(self, w):
        with torch.no_grad():
            self.w.copy_(w)

    def forward(self, token_ids):
        return self.w[token_ids]

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = nn.Parameter(
            torch.ones(d_model, device = device, dtype = dtype)
        )
    
    def use_weight(self, w):
        with torch.no_grad():
            self.g.copy_(w)

    def forward(self, x):
        d_type = x.dtype

        x = x.to(torch.float32)
        rms = (reduce(x ** 2, "... d -> ... 1", "sum") / self.d_model) + self.eps
        rms = torch.sqrt(rms)
        result = x / rms * self.g
        return result.to(d_type)


class Swiglu(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = nn.Parameter(torch.zeros(d_ff, d_model, device = device, dtype = dtype))
        self.w2 = nn.Parameter(torch.zeros(d_model, d_ff, device = device, dtype = dtype))
        self.w3 = nn.Parameter(torch.zeros(d_ff, d_model, device = device, dtype = dtype))
    
    def use_weight(self, w1_weight, w2_weight, w3_weight):
        with torch.no_grad():
            self.w1.copy_(w1_weight)
            self.w2.copy_(w2_weight)
            self.w3.copy_(w3_weight)

    def forward(self, x):
        w1x = torch.matmul(x, self.w1.t())
        w3x = torch.matmul(x, self.w3.t())
        silu = torch.sigmoid(w1x) * w1x
        temp = silu * w3x
        
        return torch.matmul(temp, self.w2.t())


class Rope(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, base: float = 10000.0, device=None):
        """
        Rotary Positional Embedding.

        Args:
            d_k (int): dimension of query/key (must be even)
            max_seq_len (int): maximum sequence length you’ll see
            base (float): rotary frequency base (Θ), default 10000
            device: where to place the precomputed buffers
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        print("dk", d_k)

        # 1. compute inv_freq for half the dims
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # 2. build a [max_seq_len, d_k//2] table of angles
        positions = torch.arange(max_seq_len, device=device).float()[:, None]
        angles = positions @ inv_freq[None, :]        # (max_seq_len, d_k//2)

        # 3. interleave sin/cos → (max_seq_len, d_k)
        sin = angles.sin().repeat_interleave(2, dim=-1)
        cos = angles.cos().repeat_interleave(2, dim=-1)

        # 4. register as buffers so they move with .to(device)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, x: torch.Tensor, token_positions: torch.LongTensor) -> torch.Tensor:
        """
        Apply RoPE to x.

        Args:
            x: tensor of shape (..., seq_len, d_k)
            token_positions: long tensor of shape (..., seq_len), values in [0, max_seq_len)
        Returns:
            the same shape as x, with rotary embeddings applied
        """
        print("x", x.size())
        # lookup sin/cos → shape (..., seq_len, d_k)
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        # rotate each pair [a, b] → [–b, a]
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        x_rot  = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

        # apply:  x * cos + rotate_half(x) * sin
        return x * cos + x_rot * sin

def softmax(in_features, dim):
    i = in_features
    i = i - torch.max(i, dim=dim, keepdim=True)[0]

    expo = torch.exp(i)
    su = torch.sum(expo, dim=dim, keepdim=True)

    return expo / su

def dot_product_attention(q, k, v, mask=None):
        d_k = q.shape[-1]

        qkt = torch.matmul(q, k.transpose(-2, -1))
        qkt = qkt / math.sqrt(d_k)

        if mask is not None:
            qkt = qkt.masked_fill(~mask, float("-inf"))

        qkt = softmax(qkt, dim=-1)
        print(qkt.shape)

        return qkt @ v

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, with_rope=False, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        self.with_rope = with_rope
        if self.with_rope:
            self.rope = Rope(self.dk, max_seq_len, theta)

    def forward(self, x, token_positions=None):
        B, T, _ = x.size()

        Q = torch.matmul(x, self.W_q)  # (B, T, d_model)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        Q = Q.view(B, T, self.num_heads, self.dk).transpose(1, 2)  # (B, H, T, dk)
        K = K.view(B, T, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.dk).transpose(1, 2)

        if self.with_rope:
            if token_positions is None:
                device = x.device
                token_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

            Q_rope = []
            K_rope = []
            for h in range(self.num_heads):
                Q_h = self.rope(Q[:, h, :, :], token_positions)  # (B, T, dk)
                K_h = self.rope(K[:, h, :, :], token_positions)  # (B, T, dk)
                Q_rope.append(Q_h)
                K_rope.append(K_h)
            
            Q = torch.stack(Q_rope, dim=1)  # (B, H, T, dk)
            K = torch.stack(K_rope, dim=1)  # (B, H, T, dk)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool().unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        attn_output = dot_product_attention(Q, K, V, mask=causal_mask)  # (B, H, T, dk)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.dk)

        output = torch.matmul(attn_output, self.W_o)
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, True, max_seq_len, theta)
        self.rms2 = RMSNorm(d_model)
        self.ffn = Swiglu(d_model, d_ff)
        self.lin = Linear(d_model, d_model)
        return
    
    def use_weight(self, weights):

        self.attn.W_q.data = weights['attn.q_proj.weight'].T
        self.attn.W_k.data = weights['attn.k_proj.weight'].T
        self.attn.W_v.data = weights['attn.v_proj.weight'].T
        self.attn.W_o.data = weights['attn.output_proj.weight'].T

        self.rms1.use_weight(weights['ln1.weight'])
        self.rms2.use_weight(weights['ln2.weight'])

        self.ffn.use_weight(
            weights['ffn.w1.weight'], 
            weights['ffn.w2.weight'], 
            weights['ffn.w3.weight']
        )

    def forward(self, in_features):
        
        normed1 = self.rms1.forward(in_features)
        sublayer1 = self.attn.forward(normed1) + in_features
        normed2 = self.rms2.forward(sublayer1)
        sublayer2 = self.ffn.forward(normed2) + sublayer1

        return sublayer2

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        layers,
        vocab_size,
    ):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.transformer_blocks = []
        self.layers = layers
        self.vocab_size = vocab_size
        for _ in range(layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta))
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
    
    def use_weight(self, weights):

        self.embed.use_weight(weights['token_embeddings.weight'])
        for i in range(self.layers):
            prefix = f"layers.{i}."

            new = {}

            for key in weights.keys():
                if prefix in key:
                    new[key[len(prefix):]] = weights[key]
                
            self.transformer_blocks[i].use_weight(new)
        
        self.norm.use_weight(weights['ln_final.weight'])
        self.linear.use_weight(weights['lm_head.weight'])
    
    def forward(self, in_indices):
        embed = self.embed.forward(in_indices)
        for i in range(self.layers):
            embed = self.transformer_blocks[i].forward(embed)
        
        normed = self.norm.forward(embed)
        linear = self.linear.forward(normed)
        return linear


