import pytest

import jax
import jax.numpy as jnp

from model.multihead_attention import MultiheadAttention, TransformerLayer

MAIN_RNG = jax.random.PRNGKey(42)


@pytest.mark.parametrize("input", [jnp.zeros((12, 200, 64))])  # (B, SeqLen, EmbedDim)
def test_multihead_attention(input):
    mh_attn = MultiheadAttention(embed_dim=64, num_heads=8)

    main_rng, init_rng = jax.random.split(MAIN_RNG, 2)
    params = mh_attn.init({"params": init_rng}, input)["params"]
    out, attention_matrix = mh_attn.apply({"params": params}, input)

    assert out.shape == input.shape and attention_matrix.shape == (
        input.shape[0],
        mh_attn.num_heads,
        input.shape[1],
        input.shape[1],
    )


# ToDo: write test
@pytest.mark.parametrize("input,train", [
    (jnp.zeros((12, 200, 64)),True)])
def test_transformer(input, train):
    tf_layer = TransformerLayer(
        embed_dim=64,
        hidden_dim=32,
        num_heads=8,
        dropout_rate=0.3,
    )

    main_rng, init_rng, dropout_init_rng = jax.random.split(MAIN_RNG, 3)

    params = tf_layer.init({"params": init_rng, "dropout": dropout_init_rng}, input, train=train)[
        "params"
    ]

    main_rng, dropout_apply_rng = jax.random.split(main_rng, 2)
    out = tf_layer.apply({"params": params}, input, rngs={"dropout": dropout_apply_rng}, train=train)

    assert out.shape == input.shape
