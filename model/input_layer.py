from flax import linen as nn
import jax.numpy as jnp
import jax

class Patcher(nn.Module):
    patch_size: int
    embed_dim: int
    method: str

    def setup(self):
        self.proj = nn.Dense(self.embed_dim,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros
        )

    def _crop(self, x) -> jnp.array:
        return None

    def _pad(self, x) -> jnp.array:
        return None
    def _create_patches(self, x) -> jnp.ndarray:
        B, H, W, C = x.shape

        #ToDo: crop/pad image to get proper shape
        x = x.reshape(B, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, -1, self.patch_size, self.patch_size, C)

        return x

    def __call__(self, x):
        # x: (B, C, W, H)
        x = self._create_patches(x) #(B, (H*W)/(P*P), P, P, C)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return self.proj(x)


#ToDo: Implement feature maps of CNN (see ViT article, hybride architecture)