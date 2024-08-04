import pytest
from PIL import Image

import jax
import jax.numpy as jnp
from jax import random

from model.vision_transformer import VisionTransformer

MAIN_RNG = jax.random.PRNGKey(42)


single_image = jnp.asarray(Image.open("../../monkey.jpeg"))
single_image = single_image.reshape(1,single_image.shape[0], single_image.shape[1], single_image.shape[2])
@pytest.mark.parametrize(
"batch",
    [
        #single_image,
        jnp.zeros((12, 100, 100, 3)),
    ]
)
def test_vit(batch):
    vit = VisionTransformer(
        patch_size=10,
        embed_dim=128,
        hidden_dim=256,
        num_heads=8,
        dropout_rate=0.1,
        num_layers=1,
        num_classes=10,
    )

    main_rng, init_rng, dropout_init_rng = random.split(MAIN_RNG, 3)
    params = vit.init({"params": init_rng, "dropout": dropout_init_rng}, batch)["params"]
    main_rng, dropout_apply_rng = random.split(main_rng, 2)

    out = vit.apply({"params": params}, batch, rngs={"dropout": dropout_apply_rng})

    assert out.shape == (batch.shape[0], vit.num_classes)

