from flax import linen as nn
import jax.numpy as jnp

from model.multihead_attention import TransformerLayer
from model.input_layer import Patcher


class VisionTransformer(nn.Module):
    patch_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    num_classes: int
    method: str = "PAD"
    dropout_rate: float = 0.1

    def setup(self):
        self.input_layer = Patcher(
            patch_size=self.patch_size, embed_dim=self.embed_dim, method=self.method
        )

        self.transformer = [TransformerLayer(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                num_heads=self.num_heads,
            )
           for _ in range(self.num_layers)]


        self.ffn_head = nn.Sequential(
            [
                nn.Dense(self.num_classes),
                nn.gelu,
            ]
        )

        self.cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim)
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + 100000, self.embed_dim), # arbitrarily high nb of params (max num of patches)
        )

    def __call__(self, batch, train): # batch: (B, C, W, H)
        x = self.input_layer(batch) # (B, Npatch, EmbedDim)
        B, n_patches, _ = x.shape
        cls_token = jnp.repeat(self.cls_token, B, 0)

        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:,:n_patches+1, :]

        for layer in self.transformer:
            x = layer(x, train=train)

        cls = x[:, 0]
        return self.ffn_head(cls)

