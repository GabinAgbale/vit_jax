import jax
import jax.numpy as jnp
from jax import random

import pytest
from PIL import Image

from model.input_layer import Patcher

MAIN_RNG = jax.random.PRNGKey(42)


single_image = jnp.asarray(Image.open("../../monkey.jpeg"))
single_image = single_image.reshape(1,single_image.shape[0], single_image.shape[1], single_image.shape[2])
@pytest.mark.parametrize(
    "batch",
    [
        single_image,
        jnp.zeros((12, 100, 100, 3)),
    ]
)
def test_input_layer(batch):
    p = Patcher(
        patch_size=20,
        emb_dim=64,
    )
    main_rng, init_rng = random.split(MAIN_RNG, 2)

    main_rng, init_rng = random.split(main_rng, 2)
    params = p.init({"params": init_rng}, batch)["params"]
    out = p.apply({'params': params}, batch)
    print('Out', out.shape)

    assert out.shape == (batch.shape[0], batch.shape[1] * batch.shape[2] // p.patch_size**2, p.emb_dim)





