"""
Training to convergence with sequential MNIST using quasi DEER
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import numpy as np
import equinox as eqx
from tqdm import tqdm

from jax import vmap
from jax.lax import scan
import wandb
import argparse
import yaml
from omegaconf import DictConfig, OmegaConf
from jax import random

import pdb

import hydra

from src.s5.dataloading import Datasets

from algs.deer import seq1d
from data.basic import load_sequential_mnist_all

class MinRNNCell(eqx.Module):
    """
    From: https://arxiv.org/pdf/1711.06788
    """
    input_weights : Float[Array, "hidden_dim input_dim"] # W_x
    input_bias: Float[Array, "hidden_dim"] # b_z
    U_z: Float[Array, "hidden_dim hidden_dim"] # U_z
    b_u: Float[Array, "hidden_dim"] # b_u
    recurrent_weights : Float[Array, "hidden_dim hidden_dim"] # U_h

    def __init__(self, key, hidden_dim, input_dim):
        k1, k2, k3 = jr.split(key, 3)
        self.input_weights = jr.normal(k1, (hidden_dim, input_dim)) / jnp.sqrt(input_dim)
        self.input_bias = jnp.zeros(hidden_dim)
        # self.recurrent_weights = jr.normal(k2, (hidden_dim, hidden_dim)) / jnp.sqrt(hidden_dim)
        self.recurrent_weights = jnp.eye(hidden_dim)  # identity initialization of hidden to hidden connection
        self.U_z = jr.normal(k3, (hidden_dim, hidden_dim)) / jnp.sqrt(hidden_dim)
        self.b_u = jnp.zeros(hidden_dim)

    def __call__(self, input, prev_state):
        z = jnp.tanh(self.input_weights @ input + self.input_bias)
        u = jax.nn.sigmoid(self.recurrent_weights @ prev_state + self.U_z @ z + self.b_u) # update gate
        state = u * prev_state + (1 - u) * z
        # output = self.output_weights @ state
        return state#, (state, output)

    def diagonal_derivative(self, input, prev_state):
        """
        Diagonal derivative of state wrt prev_state
        Should be of length hidden_dim
        Following the formula (4) in https://arxiv.org/pdf/1711.06788
        """
        z = jnp.tanh(self.input_weights @ input + self.input_bias)
        u = jax.nn.sigmoid(self.recurrent_weights @ prev_state + self.U_z @ z + self.b_u)
        return u + (prev_state - z) * u * (1-u) * jnp.diag(self.recurrent_weights)


class AugmentedGRUCell(eqx.nn.GRUCell):
    """
    eqx.nn.GRUCell with diagonal derivative method.
    """

    def __init__(self, input_size, hidden_size, use_bias=True, *, key):
        super().__init__(input_size, hidden_size, use_bias, key=key)

    def diagonal_derivative(
        self, input: jnp.ndarray, hidden: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the diagonal of the Jacobian of the cell with respect to the hidden state.

        Args:
            input: The input tensor of shape (input_size,)
            hidden: The hidden state tensor of shape (hidden_size,)
        """
        # Split the weight matrices
        w_ir, w_iz, w_in = jnp.split(self.weight_ih, 3)
        w_hr, w_hz, w_hn = jnp.split(self.weight_hh, 3)

        # Compute biases
        if self.use_bias:
            b_ir, b_iz, b_in = jnp.split(self.bias, 3)
            b_n = self.bias_n
        else:
            b_ir = b_iz = b_in = b_n = 0

        # Compute gate activations
        r_act = w_ir @ input + w_hr @ hidden + b_ir
        z_act = w_iz @ input + w_hz @ hidden + b_iz
        r = jax.nn.sigmoid(r_act)
        z = jax.nn.sigmoid(z_act)

        # Compute new gate
        rcomp = w_hn @ hidden + b_n
        n_act = w_in @ input + r * rcomp + b_in
        n = jnp.tanh(n_act)

        # Compute diagonal derivatives
        dzdh = z * (1 - z) * jnp.diag(w_hz)
        drdh = r * (1 - r) * jnp.diag(w_hr)
        dndh = (1 - n**2) * (r * jnp.diag(w_hn) + drdh * rcomp)

        # Compute diagonal Jacobian
        diag_jacobian = -dzdh * n + (1 - z) * dndh + dzdh * hidden + z

        return diag_jacobian


class GRUModel(eqx.Module):
    """
  Args
  """
    input_size: int  # Number of features of input seq (784)
    hidden_size: int  # state size for the SSM (64)
    output_size: int
    cell: eqx.Module
    out: eqx.Module
    num_iters: int
    method: str
    k : int # amount of damping
    model_type: str # "minrnn" or "gru"

    def __init__(
  self, key, input_size, hidden_size, num_iters, method='seq', k=0., model_type="minrnn",
  ):
        key1, key2 = jr.split(key)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_type = model_type
        if self.model_type == "minrnn":
            self.cell = MinRNNCell(key=key1, hidden_dim=self.hidden_size, input_dim=self.input_size)
        elif self.model_type == "gru":
            self.cell = eqx.nn.GRUCell(key=key1, hidden_size=self.hidden_size, input_size=self.input_size)
            # TODO: add diagonal derivative
            # see https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_rnn.py
        self.output_size = 10
        self.out = eqx.nn.Linear(self.hidden_size, self.output_size, key=key2)
        self.num_iters = num_iters
        self.method = method
        self.k = k

    def single_step(self, state, input):
        """
      state: jax.Array, with shape (hidden_size,)

      Keeping state, input ordering not to break the sequential scan below
      """
        if self.model_type == "minrnn":
            new_state = self.cell(input, state)  # (hidden_size,)
        elif self.model_type == "gru":
            new_state = self.cell(input, state)  # (hidden_size,)
        return (new_state, None)

    def __call__(self, inputs, states_guess=None):
        """
      Had to use an anonymous function in this scan in response to these annoying equinox / jax bugs

      https://github.com/patrick-kidger/equinox/issues/558

      https://github.com/google/jax/issues/13554

      Args:
        inputs: jax.Array, with shape (seq_len, input_size)
        flag: seq, deer
      """
        T = len(inputs)
        hidden_init = jnp.zeros((self.hidden_size,))
        if self.method == "seq":
            final_hidden, _ = jax.lax.scan(lambda *a: self.single_step(*a), hidden_init, inputs)
        elif "deer" in self.method:
            quasi_deer = "quasi" in self.method
            def model_func(state, input, model):
                return model(input, state)
            hidden_states, samp_iters = seq1d(
            model_func,
            hidden_init,
            inputs,
            self.cell,
            qmem_efficient=False,
            quasi=quasi_deer)
            jax.debug.print("# of DEER iterations: {}", samp_iters)
            final_hidden = hidden_states[-1]
        output = self.out(final_hidden)
        return output


# Define loss function (cross-entropy for classification)
@eqx.filter_jit
@eqx.filter_value_and_grad
def compute_loss(model, x, y):
  logits = jax.vmap(model)(x)  # vmap to act on a batch dimension
  one_hot_labels = jax.nn.one_hot(y, logits.shape[-1])
  loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
  return loss

@eqx.filter_jit
def compute_metrics(model, x, y):
  logits = jax.vmap(model)(x)  # vmap to act on a batch dimension
  one_hot_labels = jax.nn.one_hot(y, logits.shape[-1])
  loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
  accuracy = compute_accuracy(logits, y)
  return loss, accuracy


# Define accuracy function
def compute_accuracy(logits, labels):
  predictions = jnp.argmax(logits, axis=-1)
  return jnp.mean(predictions == labels)


# Evaluation function
# @eqx.filter_jit
def evaluate_model(model, eval_ds):
  # we could pre-compile evaluate model to avoid repeated compilation
  # but we do not jit compile inside loops
  # so we include jit in compute_metrics
  total_loss = 0.0
  num_batches = 0
  total_accuracy = 0.0

  for batch in tqdm(eval_ds, desc='Eval'):
    x = batch[0]
    y = batch[1]
    x, y = jnp.array(x.numpy()), jnp.array(y.numpy())
    x = x.swapaxes(-1, -2)  # (batch_size, input_size, seq_len)

    # Compute predictions and loss
    loss, accuracy = compute_metrics(model, x, y)
    #logits = jax.vmap(model)(x)  # vmap to act on a batch dimension
    #one_hot_labels = jax.nn.one_hot(y, logits.shape[-1])
    #loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    #accuracy = compute_accuracy(logits, y)

    # Accumulate metrics
    total_loss += loss
    total_accuracy += accuracy
    num_batches += 1

  avg_loss = total_loss / num_batches
  avg_accuracy = total_accuracy / num_batches
  return avg_accuracy, avg_loss


@eqx.filter_jit
def train_step(model, optimizer, opt_state, x, y):
  """
    Args:
      model: GRUModel
      optimizer
      opt_state: state of the optimizer
      x: pixels
      y: label
    """
  loss_value, grads = compute_loss(model, x, y)
  updates, opt_state = optimizer.update(grads, opt_state, model)
  model = eqx.apply_updates(model, updates)
  return loss_value, model, opt_state


def create_dataset(args):
  # Set randomness...
  print("[*] Setting Randomness...")
  key = random.PRNGKey(args.jax_seed)
  init_rng, train_rng = random.split(key, num=2)

  # Get dataset creation function
  create_dataset_fn = Datasets[args.dataset]

  # Dataset dependent logic
  if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
    padded = True
    if args.dataset in ["aan-classification"]:
      # Use retreival model for document matching
      retrieval = True
      print("Using retrieval model for document matching")
    else:
      retrieval = False

  else:
    padded = False
    retrieval = False

  # For speech dataset
  if args.dataset in ["speech35-classification"]:
    speech = True
    print("Will evaluate on both resolutions for speech task")
  else:
    speech = False

  # Create dataset...
  init_rng, key = random.split(init_rng, num=2)
  trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
    create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.batch_size)

  print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")
  return trainloader, valloader, testloader, aux_dataloaders


# Update the call to train_step in train_model
def train_model(model, optimizer, opt_state,
                train_ds, val_ds, test_ds,
                num_epochs, debug,
                early_stopping, early_stopping_metric="val_loss", patience=5, min_delta=1e-4):
    best_metric = float("inf") if early_stopping_metric == "val_loss" else float("-inf")
    no_improvement_epochs = 0

    if debug:
        num_epochs = [0]
        all_batches = [next(iter(train_ds))]
        all_val_batches = [next(iter(val_ds))]
        all_test_batches = [next(iter(test_ds))]

    else:
        num_epochs = range(num_epochs)
        all_batches = train_ds
        all_val_batches = val_ds
        all_test_batches = test_ds

    num_batches = 0

    for epoch in tqdm(num_epochs, desc="Training epoch"):
        total_loss = 0.0
        # Training loop
        for batch in tqdm(all_batches):
            x = batch[0]
            y = batch[1]
            x, y = jnp.array(x.numpy()), jnp.array(y.numpy())
            x = jnp.swapaxes(x, -1, -2)  # (batch_size, input_size, seq_len)
            loss_value, model, opt_state = train_step(model, optimizer, opt_state, x, y)  # Pass model explicitly

            if wandb.run is not None:
                metrics = {"train/train_batch_loss": loss_value}
                wandb.log(metrics)

            total_loss += loss_value
            num_batches += 1

        # Log training metrics
        avg_loss = total_loss / num_batches

        if wandb.run is not None:
            metrics = {"train/train_loss": avg_loss,
                 "train/epoch": epoch}
            wandb.log(metrics)

        # Evaluate after each epoch
        val_accuracy, val_loss = evaluate_model(model, all_val_batches)
        jax.block_until_ready(val_accuracy)
        jax.block_until_ready(val_loss)

        if wandb.run is not None:
            metrics = {"val/val_loss": val_loss,
                 "val/epoch": epoch,
                 "val/accuracy": val_accuracy}
            wandb.log(metrics)

        # Early stopping logic
        if early_stopping:
            current_metric = val_loss if early_stopping_metric == "val_loss" else val_accuracy
            if (early_stopping_metric == "val_loss" and current_metric < best_metric - min_delta) or \
        (early_stopping_metric == "val_accuracy" and current_metric > best_metric + min_delta):
                best_metric = current_metric
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        del val_accuracy, val_loss  # Free memory after logging

    # TODO: update to best epoch
    # Log full test
    print(f"[*] Evaluating on test set...")
    test_accuracy, test_loss = evaluate_model(model, all_test_batches)

    if wandb.run is not None:
        metrics = {"test/test_loss": test_loss,
               "test/epoch": epoch,
               "test/accuracy": test_accuracy}
        wandb.log(metrics)
    return model, opt_state, test_loss, test_accuracy


@hydra.main(config_path=".", config_name="experiment")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, config=dict(cfg), mode="offline")

    # Load datasets
    trainloader, valloader, testloader = load_sequential_mnist_all(
        batch_size=cfg.batch_size, val_split=0.1, seed=0
    )

    # Initialize model
    model = GRUModel(
        jr.PRNGKey(0),
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_iters=cfg.num_iters,
        method=cfg.method,
        k=cfg.k,
        model_type=cfg.model_type,
    )

    # Initialize optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(cfg.learning_rate, b1=0.9, b2=0.999, weight_decay=0.0),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Train and evaluate model
    _ = train_model(
        model,
        optim,
        opt_state,
        trainloader,
        valloader,
        testloader,
        cfg.num_epochs,
        cfg.debug,
        cfg.early_stopping,
        cfg.early_stopping_metric,
        cfg.early_stopping_patience,
        cfg.early_stopping_min_delta,
    )

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
