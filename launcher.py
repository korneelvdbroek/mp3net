# !/usr/bin/env python

from __future__ import print_function

import datetime
import subprocess
import logging

import tensorflow as tf

from model.progressivetrainer import ProgressiveTrainer
from utils import gspath


logger = logging.getLogger('MP3net')
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s.%(msecs)03d   : %(levelname).1s %(pathname)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def setup_tpu():
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tpu_ip = tpu.cluster_spec().as_dict()['worker']
    print('Running on TPU ', tpu_ip)
  except ValueError:
    raise BaseException(
      'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

  tf.config.experimental_connect_to_cluster(tpu)
  tpu_topology = tf.tpu.experimental.initialize_tpu_system(tpu)  # prints info to screen
  tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
  return tpu_strategy


def get_training_dir(args, confirm=True):
  print("Determining training directory...")

  # figure out the training directory
  training_dir = None
  if args.training_sub_dir is not None:
    # user specified a tag to use
    training_dir = gspath.join(args.training_base_dir, args.training_sub_dir)

    if not gspath.is_dir(training_dir):
      raise ValueError("Training directory {} does not exist... exiting".format(training_dir))

    if confirm:
      answer = input("Do you want to reuse the training directory {}? ".format(training_dir))
      if answer.lower()[0] == "y":
        print("Ok, re-using directory {}".format(training_dir))
      else:
        args.training_sub_dir = None

  if args.training_sub_dir is None:
    # creating new training directory
    args.training_sub_dir = "train_{0}".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    training_dir = gspath.join(args.training_base_dir, args.training_sub_dir)
    print("Training in new directory {}".format(training_dir))

    # Make training dir
    if not gspath.is_dir(training_dir):
      gspath.mkdir(training_dir)

  print(f"  Training directory is {training_dir}")
  return training_dir


def execute(args, strategy=None):
  # set up device strategy
  if args.runtime_tpu:
    # rely on the externally defined tpu_strategy
    if strategy is None:
      raise Exception("Strategy should be defined for TPU run")
    print("Running on TPU")
  elif args.mode == 'eval':
    # # make only CPU visible for evaluation loop (otherwise it tries to grab the cuda already in use)
    # physical_devices = tf.config.list_physical_devices('CPU')
    # tf.config.set_visible_devices(physical_devices)
    # device = "/cpu:0"

    device = "/gpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device)
    print("Running on {}".format(device))
  else:
    device = "/gpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device)
    print("Running on {}".format(device))

  model_trainer = ProgressiveTrainer(strategy, args)

  # start tensorboard
  tensorboard_process = None
  try:
    if args.runtime_launch_tensorboard:
      print("Launching tensorboard...")
      exec_str = "tensorboard --logdir={0}".format(args.training_dir)
      print('  ' + exec_str)
      tensorboard_process = subprocess.Popen(exec_str)
      print(f'  PID = {tensorboard_process.pid}')

    # decode running mode
    if args.mode == 'train':
      # Save args
      filepath = gspath.join(args.training_dir, 'args_{0}.txt'.format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")))
      data_str = '\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
      print('Writing arguments to {}'.format(filepath))
      gspath.write(filepath, data_str)

      model_trainer.progressive_training()

    elif args.mode == 'eval':
      model_trainer.evaluation_loop()

    elif args.mode == 'infer':
      model_trainer.inference_loop()

    else:
      raise NotImplementedError()

  finally:
    if tensorboard_process is not None:
      # kill tensorboard
      subprocess.call(['taskkill', '/F', '/T', '/PID', str(tensorboard_process.pid)])


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'infer', 'eval'])
  parser.add_argument('--training_base_dir', type=str,
      help='Base directory for training')
  parser.add_argument('--training_sub_dir', type=str,
      help='Tag of existing training directory to be used')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_shuffle_buffer_size', type=int,
      help='Buffer with data from which we sample randomly')

  runtime_args = parser.add_argument_group('Runtime')
  runtime_args.add_argument('--launch_tensorboard', action='store_true', dest='runtime_launch_tensorboard',
      help='If set, run tensorboard during training')
  runtime_args.add_argument('--tpu', action='store_true', dest='runtime_tpu',
      help='Run on TPU')
  runtime_args.add_argument('--tpu_profiling', action='store_true', dest='tpu_profiling',
      help='Run profiling on TPU during training run')

  train_args = parser.add_argument_group('Train')
  # Critic updates per generator update
  # BigGAN finds 2 optimal, StyleGAN even works with 1
  # If we take it too small, gradients might push the generator through the submanifold with the real distribution
  #   since the discriminator does not have the time to adjust (see https://arxiv.org/abs/1801.04406)
  # so especially, once we are converging, a higher N_DISCR might be warranted
  train_args.add_argument('--n_discr', type=int,
      help='Number of discriminator updates for each generator update')
  train_args.add_argument('--batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_checkpoint_freq', type=int,
      help='Number of discriminator optimization steps to save the model')

  eval_args = parser.add_argument_group('Eval')
  eval_args.add_argument('--summary_dir', type=str,
      help='Summary directory')
  eval_args.add_argument('--summary_freq', type=int,
      help='Number of discriminator optimization steps to write a summary')

  infer_args = parser.add_argument_group('Infer')
  infer_args.add_argument('--infer_dir', type=str,
      help='Directory for inference')

  parser.set_defaults(
    data_dir=None,
    batch_size=8,
    train_checkpoint_freq=100,
    summary_freq=500,
    data_shuffle_buffer_size=2**12,
    runtime_launch_tensorboard=False,
    runtime_tpu=False,
    tpu_profiling=False)

  args = parser.parse_args()

  setattr(args, "training_dir", get_training_dir(args, confirm=(args.mode == 'train')))
  setattr(args, "summary_dir", f"{args.training_dir}/summary/")

  execute(args)
