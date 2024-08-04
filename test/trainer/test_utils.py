import pytest

from trainer.utils import *
from trainer.trainer import Trainer
from data.loader import cifar_loaders

@pytest.fixture
def vit_hparams():
    return  {
    'patch_size': 4,
    'embed_dim': 128,
    'hidden_dim': 256,
    'num_heads': 8,
    'dropout_rate': 0.1,
    'num_layers': 1,
    'num_classes': 10
    }

train_loader, eval_loader, test_loader = cifar_loaders()

@pytest.mark.parametrize(
    "batch,train_loader,eval_loader",
    [
        (next(iter(train_loader)), train_loader, eval_loader),
    ]
)
def test_train_step(batch, train_loader, eval_loader, vit_hparams):
    trainer = Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        log_dir="/Users/gabin/PycharmProjects/jax_ViT/logs",
        batch_sample=next(iter(train_loader))[0],
        num_epochs=2,
        **vit_hparams,
    )

    trainer.init_train_state()

    print("Start training...")
    state, rng, loss, acc = trainer.train_step(batch=batch,
                                               rng=trainer.rng)

    print("Finished training...")