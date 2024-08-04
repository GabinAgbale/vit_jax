import math

import jax.numpy as jnp
from flax import linen as nn


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


# ToDo: handle device, add dropout
class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)  # (B, S, 3*d)

        qkv = qkv.reshape(
            batch_size, seq_length, self.num_heads, -1
        )  # (B, S, H, 3*d//H)
        qkv = qkv.transpose(0, 2, 1, 3)  # (B, H, S, 3*d//H)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # q, k, v: (B, H, S, d//H)

        # Determine value outputs
        values, attention = scaled_dot_product(
            q, k, v, mask=mask
        )  # value: (B, H, SeqLen, d//H), attention: (B, H, SeqLen, SeqLen)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, H , d//H]

        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


class TransformerLayer(nn.Module):
    embed_dim: int
    hidden_dim: int
    dropout_rate: float
    num_heads: int

    def setup(self):
        self.attn = MultiheadAttention(self.embed_dim, self.num_heads)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        self.ffn = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            nn.Dense(self.embed_dim),
        ]

    def __call__(self, x, train, mask=None):

        attn_out, _ = self.attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic= not train)
        x = self.norm1(x)

        ffn_out = jnp.copy(x)
        for layer in self.ffn:
            ffn_out = layer(ffn_out) if not isinstance(layer, nn.Dropout) else layer(ffn_out, deterministic= not train)

        x = x + self.dropout(x, deterministic=not train)

        return self.norm2(x)
