# -*- coding: utf-8 -*-
"""deer_gru_seq_mnist.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fCaaXaN5Febh7zySGuEIlPhtxQWRovLH

# Applying DEER to GRUs on sequential MNIST

TODOs:
* We are often running out of RAM: we probably need quasi
* will need a custom diagonal derivative for the equinox Gru
* still need to try warm starting

## Set up run
python example.py -b experiment.yaml

# flags to avoid redundant init

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

"""

# ! pip install equinox

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import numpy as np
import equinox as eqx
import pdb
from tqdm import tqdm

from jax import vmap
from jax.lax import scan
import wandb
import argparse
import yaml
from omegaconf import OmegaConf
from jax import random

from src.s5.dataloading import Datasets


def elk_alg(
  f,
  initial_state,
  states_guess,
  drivers,
  num_iters=10,  # controls number of iteration
  quasi=False,
):
  """
    Currently is DEER

    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (L-1, DIM)
      drivers, jax.Array, (L-1,N_noise)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:L-1] be the states, and e[0:L-2] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{L-2} ----> h_{L-1}
              |       |                   |          |
              e1      e2       ..... e_{L-2}      e_{L-1}
    """
  DIM = len(initial_state)
  L = len(drivers)

  @jax.vmap
  def full_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j

  @jax.vmap
  def diag_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

  @jax.jit
  def _step(states, args):
    # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
    fs = vmap(f)(states[:-1], drivers[1:])  # get the next
    # pdb.set_trace()
    # Jfs are the Jacobians (what is going with the tuples rn)
    Jfs = vmap(jax.jacrev(f, argnums=0))(
      states[:-1], drivers[1:]
    )

    # Compute the As and bs from fs and Jfs
    if quasi:
      As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
      bs = fs - As * states[:-1]

    else:
      As = Jfs
      bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

    # initial_state is h0
    b0 = f(initial_state, drivers[0])  # h1
    A0 = jnp.zeros_like(As[0])
    A = jnp.concatenate(
      [A0[jnp.newaxis, :], As]
    )  # (L, D, D) [or (L, D) for quasi]
    b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (L, D)
    if quasi:
      binary_op = diag_mat_operator
    else:
      binary_op = full_mat_operator

    # run appropriate parallel alg
    _, new_states = jax.lax.associative_scan(binary_op, (A, b))  # a forward pass, but uses linearized dynamics
    new_states = jnp.nan_to_num(new_states)  # zero out nans
    return new_states, None

  final_state, _ = scan(_step, states_guess, None, length=num_iters)
  return final_state[-1]


# Load MNIST and flatten for Sequential MNIST task with tfds.as_numpy
def load_sequential_mnist(split, batch_size):
  ds = tfds.load("mnist", split=split, as_supervised=True)
  ds = ds.shuffle(1024).map(lambda x, y: (tf.reshape(x, [28 * 28]), y))  # Flatten each image to 1D sequence
  ds = ds.map(lambda x, y: (tf.transpose(tf.expand_dims(x, axis=0)), y))  # Add sequence length dimension
  ds = ds.batch(batch_size).prefetch(10)
  return tfds.as_numpy(ds)  # Convert entire dataset to NumPy arrays


def load_sequential_mnist_v2(split, batch_size, validation_split=0.1):
  # Load the full dataset
  ds = tfds.load("mnist", split="train" if split == "train" else split, as_supervised=True)

  if split == "train":
    # Split into training and validation sets
    total_size = tf.data.experimental.cardinality(ds).numpy()
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    # Process the training dataset
    train_ds = train_ds.shuffle(1024).map(process_fn)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Process the validation dataset
    val_ds = val_ds.map(process_fn)
    val_ds = val_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(train_ds), tfds.as_numpy(val_ds)

  else:
    # For test dataset, process normally
    total_size = tf.data.experimental.cardinality(ds).numpy()
    ds = ds.map(process_fn)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)

def process_fn(x, y):
    return (tf.transpose(tf.expand_dims(tf.reshape(x, [28 * 28]), axis=0)), y)


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

  def __init__(
    self, key, input_size, hidden_size, num_iters, method='seq'
  ):
    key1, key2 = jr.split(key)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.cell = eqx.nn.GRUCell(self.input_size, self.hidden_size, key=key1)
    self.output_size = 10
    self.out = eqx.nn.Linear(self.hidden_size, self.output_size, key=key2)
    self.num_iters = num_iters
    self.method = method

  def single_step(self, state, input):
    """
        state: jax.Array, with shape (hidden_size,)
        """
    new_state = self.cell(input, state)  # (hidden_size,)
    return (new_state, None)

  def __call__(self, inputs):
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
      final_hidden, _ = jax.lax.scan(
        lambda *a: self.single_step(*a), hidden_init, inputs
      )
    elif self.method == "deer":
      final_hidden = elk_alg(
        f=lambda state, input: self.single_step(state, input)[0],
        initial_state=hidden_init,
        states_guess=jnp.zeros((T, self.hidden_size)),  # TODO: we will want to warm start
        drivers=inputs,
        num_iters=self.num_iters,
        quasi=False)
    # pdb.set_trace()
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
@eqx.filter_jit
def evaluate_model(model, eval_ds):
  total_loss = 0.0
  num_batches = 0
  total_accuracy = 0.0

  for batch in tqdm(eval_ds, desc='Eval'):
    # x, y, _ = batch
    #x, y = batch
    x = batch[0]
    y = batch[1]
    x, y = jnp.array(x), jnp.array(y)

    # Compute predictions and loss
    loss, accuracy = compute_metrics(model, x, y)
    # pre-compile evaluate model to avoid repeated compilation
    # do not jit compile inside loops
    #logits = jax.vmap(model)(x)  # vmap to act on a batch dimension
    #one_hot_labels = jax.nn.one_hot(y, logits.shape[-1])
    #loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    #accuracy = compute_accuracy(logits, y)

    # Accumulate metrics
    total_loss += loss
    total_accuracy += accuracy
    num_batches += 1

  breakpoint()
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
  updates, opt_state = optimizer.update(grads, opt_state)
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

  total_loss = 0.0
  total_acc = 0.0
  num_batches = 0

  for epoch in tqdm(num_epochs, desc="Training epoch"):
    # Training loop
    for batch in tqdm(all_batches):
      x = batch[0]
      y = batch[1]
      # x, y, _ = batch
      #x, y = batch
      x, y = jnp.array(x), jnp.array(y)
      loss_value, model, opt_state = train_step(model, optimizer, opt_state, x, y)  # Pass model explicitly

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
    breakpoint()
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


"""## Train model"""


def main(args):
  # Let's try to train a model:

  # TODO: seed everything equivalent in jax
  input_size = 1
  hidden_size = args.hidden_size  # 256
  learning_rate = args.learning_rate
  num_iters = args.num_iters
  method = args.method
  debug = args.debug
  wandb_project = args.wandb_project

  early_stopping = args.early_stopping
  early_stopping_metric = args.early_stopping_metric
  patience = args.early_stopping_patience
  min_delta = args.early_stopping_min_delta

  # Start a wandb run
  if args.use_wandb:
    wandb.init(project=wandb_project, config=dict(args))

  # Load datasets
  trainloader, valloader = load_sequential_mnist_v2(split="train", batch_size=args.batch_size)
  testloader = load_sequential_mnist_v2(split="test", batch_size=args.batch_size)
  # trainloader = load_sequential_mnist(split="train", batch_size=args.batch_size)
  # valloader = load_sequential_mnist(split="test", batch_size=args.batch_size)
  # testloader = load_sequential_mnist(split="test", batch_size=args.batch_size)
  # preallocation issues?
  #trainloader, valloader, testloader, aux_dataloaders = create_dataset(args)

  # Initialize and train the model:
  model = GRUModel(jr.PRNGKey(0), input_size, hidden_size, num_iters, method)
  optim = optax.adam(learning_rate)
  opt_state = optim.init(eqx.filter(model, eqx.is_array))

  # train and eval model
  _ = train_model(model, optim, opt_state,
                  trainloader, valloader, testloader,
                  args.num_epochs, debug,
                  early_stopping, early_stopping_metric, patience, min_delta)

  if args.use_wandb:
    wandb.finish()
  return


# Old debugging code:
class Arguments:
  num_epochs = 1
  num_iters = 10
  method = 'seq'  # 'deer'
  use_wandb = False
  debug = True
  batch_size = 32  # 256
  learning_rate = 0.001
  hidden_size = 16
  seed = 0


if __name__ == "__main__":
  # export XLA_PYTHON_CLIENT_PREALLOCATE=false
  # export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  # args = Arguments()
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("-b", "--config", required=True, help="Path to the experiment YAML configuration file")
  args, unknown = parser.parse_known_args()

  # Load the YAML configuration file
  with open(args.config, "r") as f:
    config = yaml.safe_load(f)
  config = OmegaConf.create(config)
  main(config)

