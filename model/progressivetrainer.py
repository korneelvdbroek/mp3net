# !/usr/bin/env python

"""Trainer that takes the distributed model driver through the different progressive stages"""

from __future__ import print_function

import re
import time
import datetime

from model import mp3net
from model.modeldriver import DistributedModelDriver
from utils import gspath


class ProgressiveTrainer:
  def __init__(self, strategy, args):
    """Train specGAN

    :param strategy:
    :param args:             user arguments
    """
    self.strategy = strategy
    self.args = args

    # set up model_factory
    self.model_factory = mp3net.MP3netFactory()

    print("ProGAN training:")
    for stage, (blocks_n, freq_n) in enumerate(zip(self.model_factory.blocks_n, self.model_factory.freq_n)):
      print("  Stage {0}: {1:5d}x{2:5d}".format(stage, blocks_n, freq_n))

  def progressive_training(self):
    stage_start = self._find_latest_stage()
    if stage_start is not None:
      print("Restoring stage {} from checkpoint file".format(stage_start))
    else:
      print("Starting anew from stage 0")
      stage_start = 0

    model_weights = None
    for stage in range(stage_start, self.model_factory.stages):
      print()
      if self.model_factory.stage_total_songs[stage] == 0:
        print("======= Empty stage {0} =======".format(stage))
      else:
        blocks_n = self.model_factory.blocks_n[stage]
        freq_n = self.model_factory.freq_n[stage]
        print("======= Entering stage {0} =======".format(stage))
        print(f'Stage {stage}:')
        print("  * resolution            = {0:d} x {1:d}".format(blocks_n, freq_n))
        print("  * total number of songs = {0:,}".format(self.model_factory.stage_total_songs[stage]))
        print()

        # set up summary writer (different writer for different stage, since different tensors etc)
        summary_dir = gspath.join(self.args.summary_dir, f"stage-{stage}")

        # 2. start model driver
        model_driver = DistributedModelDriver(self.strategy, self.model_factory, stage,
                                              mode='train', summary_dir=summary_dir, args=self.args)
        if model_weights is None:
          model_driver.load_latest_checkpoint()
        else:
          model_driver.load_from_model_weights(model_weights, self.model_factory.map_layer_names_for_model_growth)

        model_driver.print_model_summary()

        # 3. run model
        data_files = self.find_data_files(stage, self.args.data_dir, self.model_factory)
        model_weights = model_driver.training_loop(data_files)

        print("======= Exiting stage {} =======".format(stage))

  def evaluation_loop(self):
    current_stage, current_checkpoint_no = self._find_latest_checkpoint()
    print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Most recent is stage-{current_stage} and checkpoint-{current_checkpoint_no}...")

    # loop over different stages
    while True:
      print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Entering stage-{current_stage}...")

      print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Initializing model and data for stage-{current_stage}...")

      # set up summary writer (different writer for different stage, since different tensors etc)
      summary_dir = gspath.join(self.args.summary_dir, f"stage-{current_stage}")

      model_driver = DistributedModelDriver(self.strategy, self.model_factory, current_stage,
                                            mode='eval', summary_dir=summary_dir, args=self.args)
      data_files = self.find_data_files(current_stage, self.args.data_dir, self.model_factory)
      dataset_iter = model_driver.get_distributed_dataset(data_files)

      current_checkpoint_no = max(0, current_checkpoint_no - 1)

      # eternal evaluation loop of writing summaries for stage = current_stage
      while True:
        new_stage, new_checkpoint_no = self._find_latest_checkpoint()

        if new_stage > current_stage:
          print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: New stage found...")
          current_stage = new_stage
          current_checkpoint_no = new_checkpoint_no - 1  # minus 1 since it's still untreated!
          break

        if new_stage > current_stage or (new_stage == current_stage and new_checkpoint_no > current_checkpoint_no):
          print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: New checkpoint found...")
          # wait a bit more, to make sure all checkpoint files are saved to disk

          model_driver.load_latest_checkpoint()
          model_driver.evaluation_loop(dataset_iter)
          print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Waiting for new checkpoint...")

        current_stage = new_stage
        current_checkpoint_no = new_checkpoint_no

        time.sleep(5)

  def inference_loop(self):
    stage = self._find_latest_stage()
    if stage is not None:
      print("Restoring stage {} from checkpoint file".format(stage))
    else:
      raise ValueError("No checkpoint file found")

    blocks_n = self.model_factory.blocks_n[stage]
    freq_n = self.model_factory.freq_n[stage]
    print("======= Entering stage {0} =======".format(stage))
    print(f'Stage {stage}:')
    print("  * resolution            = {0:d} x {1:d}".format(blocks_n, freq_n))
    print("  * total number of songs = {0:,}".format(self.model_factory.stage_total_songs[stage]))
    print()

    # 1. start model driver
    model_driver = DistributedModelDriver(self.strategy, self.model_factory, stage, mode='infer', args=self.args)
    model_driver.load_latest_checkpoint()
    model_driver.print_model_summary()

    # 2. run model
    model_driver.inference_loop()

    print("======= Exiting stage {} =======".format(stage))

  def _find_latest_stage(self):
    """
    :return:  number of latest saved model stage, or None is no checkpoints in any stages were found
    """
    checkpoint_file_path = gspath.join(self.args.training_dir, "stage-*", "stage-*-ckpt-*.data*")
    checkpoint_files = gspath.findall(checkpoint_file_path)
    stages = [int(re.search(r'stage-(\d+)-ckpt-\d+\.data', checkpoint_file).group(1)) for checkpoint_file in checkpoint_files
              if re.search(r'stage-(\d+)-ckpt-\d+\.data', checkpoint_file) is not None]

    return max(stages) if stages else None

  def _find_latest_checkpoint(self):
    """
    :return: tuple (stage, checkpoint)
             if no checkpoint in any stage is found, then (model_start_stage, 0) is returned
    """
    stage = self._find_latest_stage()
    if stage is not None:
      checkpoint_file_path = gspath.join(self.args.training_dir, f"stage-{stage}", f"stage-{stage}-ckpt-*.data*")
      checkpoint_files = gspath.findall(checkpoint_file_path)
      checkpoint_no = [int(re.search(r'stage-\d+-ckpt-(\d+)\.data', checkpoint_file).group(1)) for checkpoint_file in checkpoint_files
                       if re.search(r'stage-\d+-ckpt-(\d+)\.data', checkpoint_file) is not None]
    else:
      stage = 0
      while self.model_factory.stage_total_songs[stage] == 0 and stage < self.model_factory.stages:
        stage += 1
      checkpoint_no = [0]

    return stage, max(checkpoint_no)

  @staticmethod
  def find_data_files(training_stage, data_dir, model_factory):
    freq_n = model_factory.freq_n[training_stage]
    channels_n = model_factory.channels_n[training_stage]

    # find appropriate pre-processed data files
    file_pattern = f'*_sr{model_factory.sample_rate}_Nx{freq_n}x{channels_n}.tfrecord'
    input_file_path = gspath.join(data_dir, file_pattern)
    input_files = gspath.findall(input_file_path)[0:]
    if len(input_files) == 0:
      raise ValueError(f'Did not find any preprocessed file {file_pattern} in directory {data_dir}')
    print("Found {0} data files in {1}: ".format(len(input_files), data_dir))
    for input_filepath in input_files:
      print("  {}".format(gspath.split(input_filepath)[1]))

    return input_files
