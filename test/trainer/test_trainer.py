import pytest
from flax.training import train_state

from model.vision_transformer import VisionTransformer
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
    "train_loader,eval_loader",
    [
        (train_loader, eval_loader),
    ]
)
def test_trainer(train_loader, eval_loader, vit_hparams):
    trainer = Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        log_dir="/Users/gabin/PycharmProjects/jax_ViT/logs",
        batch_sample=next(iter(train_loader))[0],
        num_epochs=2,
        **vit_hparams,
    )

    assert isinstance(trainer, Trainer)


@pytest.mark.parametrize(
    "train_loader,eval_loader",
    [
        (train_loader, eval_loader),
    ]
)
def test_init_train_state(train_loader, eval_loader, vit_hparams):
    trainer = Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        log_dir="/Users/gabin/PycharmProjects/jax_ViT/logs",
        batch_sample=next(iter(train_loader))[0],
        num_epochs=2,
        **vit_hparams,
    )

    trainer.init_train_state()

    assert isinstance(trainer.state, train_state.TrainState)


@pytest.mark.parametrize(
    "train_loader,eval_loader",
    [
        (train_loader, eval_loader),
    ]
)
def test_train_epoch(train_loader, eval_loader, vit_hparams):
    trainer = Trainer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        log_dir="/Users/gabin/PycharmProjects/jax_ViT/logs",
        batch_sample=next(iter(train_loader))[0],
        num_epochs=2,
        **vit_hparams,
    )

    trainer.train_epoch()