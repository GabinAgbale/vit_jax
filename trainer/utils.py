import jax
import jax.numpy as jnp

import optax


def compute_cross_entropy_loss_and_acc(model, params, batch, rng, train):
    rng, dropout_rng = jax.random.split(rng)
    imgs, targets = batch

    logits = model.apply(
        {"params": params}, imgs, train=train, rngs={"dropout": dropout_rng}
    )

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    acc = (logits.argmax(axis=-1) == targets).mean()

    return loss, (rng, acc)

@jax.jit
def train_step(state, model, rng, batch):
    loss_fn = lambda params: compute_cross_entropy_loss_and_acc(model, params, batch, rng, train=True)
    (loss, (rng, acc)), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grad)
    return state, rng, loss, acc

@jax.jit
def eval_step(state, model, rng, batch):
    _, (rng, acc) = compute_cross_entropy_loss_and_acc(model, state.params, batch, rng, train=False)
    return rng, acc



