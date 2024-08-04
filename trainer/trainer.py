import os
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import jax
import optax
from jax import random
from flax.training import checkpoints, train_state

from torch.utils.tensorboard import SummaryWriter

from model.vision_transformer import VisionTransformer




class Trainer:
    def __init__(
        self,
        train_loader,
        eval_loader,
        log_dir,
        batch_sample,
        num_epochs,
        optimizer="ADAM",
        lr_scheduler="LINEAR",
        warmup_rate=0.1,
        init_lr=1e-3,
        max_lr=1e-2,
        weight_decay=1e-2,
        seed=42,
        **vit_hparams
    ):
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.nb_step_per_epoch = len(train_loader)

        self.num_epochs = num_epochs

        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.warmup_rate = warmup_rate
        self.init_lr = init_lr
        self.max_lr = max_lr

        self.weight_decay = weight_decay

        self.seed = seed
        self.rng = random.PRNGKey(self.seed)

        self.vit = VisionTransformer(**vit_hparams)

        self.log_dir = log_dir
        self.logger = SummaryWriter(log_dir=self.log_dir)

        self.batch_sample = batch_sample

        self.create_jitted_functions()
        self.init_model()

    def init_model(self):
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        self.init_params = self.vit.init(
            {"params": init_rng, "dropout": dropout_init_rng},
            self.batch_sample,
                  train=True,
        )["params"]

        # To notify when first training step
        self.state = None

    def setup_scheduler(self):
        match self.lr_scheduler:
            case "WARMUP":
                assert (
                        self.init_lr is not None
                        and self.max_lr is not None
                        and self.warmup_rate is not None
                )
                assert self.init_lr < self.max_lr

                warmup = optax.linear_schedule(init_value=self.init_lr,
                                               end_value=self.max_lr,
                                               transition_steps=1)
                cosine_decay = optax.cosine_decay_schedule(init_value=self.max_lr,
                                                           decay_steps=1,
                                                           alpha=self.init_lr)
                scheduler = optax.join_schedules([warmup, cosine_decay],
                                                 boundaries=[
                                                     int(self.num_epochs * self.nb_step_per_epoch * self.warmup_rate)])

            case "LINEAR":
                assert (self.init_lr is not None) and (self.max_lr is not None)
                assert self.init_lr < self.max_lr

                scheduler = optax.linear_schedule(self.init_lr, self.max_lr, self.num_epochs)

            case "CONSTANT":
                scheduler = optax.constant_schedule(self.init_lr)

            case _:
                raise KeyError(f"Unknown scheduler: {self.lr_scheduler}, choose from WARMUP, LINEAR, CONSTANT")

        return scheduler

    def setup_optimizer(self, scheduler):
        g_clip = optax.clip_by_global_norm(1.0)  # clip gradient norm at 1
        match self.optimizer:
            case "ADAM":
                optimizer = optax.chain(
                    g_clip,
                    optax.adam(scheduler, b1=0.9, b2=0.999),
                )

            case "ADAMW":
                optimizer = optax.chain(
                    g_clip,
                    optax.adamw(scheduler, b1=0.9, b2=0.999,
                                weight_decay=self.weight_decay),
                )

            case "SGD":
                optimizer = optax.chain(
                    g_clip,
                    optax.sgd(scheduler)
                )

            case _:
                raise KeyError(f"Unknown optimizer: {self.optimizer}, choose from ADAM, ADAMW, SGD")

        return optimizer


    def init_train_state(self):

        # Initialize LR scheduler
        scheduler = self.setup_scheduler()

        # Initialize optimizer
        optimizer = self.setup_optimizer(scheduler)

        self.state = train_state.TrainState.create(
            apply_fn=self.vit.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer)


    def create_jitted_functions(self):

        def compute_cross_entropy_loss_and_acc(params, batch, rng, train):
            rng, dropout_rng = jax.random.split(rng)
            imgs, targets = batch

            logits = self.vit.apply(
                {"params": params}, imgs, train=train, rngs={"dropout": dropout_rng}
            )

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            acc = (logits.argmax(axis=-1) == targets).mean()

            return loss, (rng, acc)

        def train_step(rng, batch):
            loss_fn = lambda params: compute_cross_entropy_loss_and_acc(params, batch, rng, train=True)
            (loss, (rng, acc)), grad = jax.value_and_grad(loss_fn, has_aux=True)(self.state.params)

            state = self.state.apply_gradients(grad)
            return state, rng, loss, acc

        def eval_step(model, rng, batch):
            _, (rng, acc) = compute_cross_entropy_loss_and_acc(model, state.params, batch, rng, train=False)
            return rng, acc

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)


    def train_epoch(self, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(self.train_loader, desc='Training', leave=False):
            self.state, self.rng, loss, acc = self.train_step(batch,
                                                         model=self.vit,
                                                         state=self.state,
                                                         rng=self.rng)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/' + key, avg_val, global_step=epoch)


    def save_model(self, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target=self.state.params,
                                    step=step,
                                    overwrite=True)

    def eval_model(self):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in self.eval_loader:
            self.rng, acc = self.eval_step(batch, model=self.vit, state=self.state, rng=self.rng)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def train_model(self):
        self.init_train_state()

        best_eval = 0.0
        for epoch_idx in tqdm(range(1, self.num_epochs + 1)):
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model()
                self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def load_model(self, pretrained=False):
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.log_dir, f'ViT.ckpt'), target=None)


        self.state = train_state.TrainState.create(
            apply_fn=self.vit.apply,
            params=params,
            tx=self.state.tx if self.state else self.setup_optimizer(self.setup_scheduler())  # Default optimizer
        )





