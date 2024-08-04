import matplotlib
import pytest
import jax.numpy as jnp
from PIL import Image
import jax

from model.input_layer import InputLayer
matplotlib.use('module://backend_interagg')


single_image = jnp.asarray(Image.open("../monkey.jpeg").convert("L"))
single_image = single_image.reshape(1, 1,single_image.shape[0], single_image.shape[1])
@pytest.mark.parametrize(
    "batch",
    [
        single_image,
        jnp.zeros((12, 3, 70, 70)),
    ]
)
def test_patcher(batch):
    p_size = 100
    patcher = InputLayer(
        patch_size=70,
        emb_dim=64,
    )

    variable = patcher.init(jax.random.key(0), batch)
    patches = patcher.apply(variable)




