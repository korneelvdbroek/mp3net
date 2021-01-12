# !/usr/bin/env python

"""Training and inference loops to run model in distributed GPU & TPU environments"""

from __future__ import print_function

import os
import datetime
import subprocess  # needed for TPU profiling

import tensorflow as tf
import numpy as np
import scipy as sc
import scipy.linalg
import matplotlib.cm as cm

from tensorboardX import SummaryWriter

from model import mp3net, dataloader
from model.audiorepresentation import AudioRepresentation
import utils.audio_utils as audio_utils
from utils import gspath

import imageio


class DistributedModelDriver:
  def __init__(self, strategy, model_factory: mp3net.MP3netFactory, stage, mode, summary_dir, args):
    """Model Driver owns the actual TranceModel object and takes care of running it on a distributed environment

    :param strategy:
    :param args:             user arguments
    """
    self.strategy = strategy
    self.args = args
    self.global_batch_size = self.args.batch_size

    with self.strategy.scope():
      # use bfloat16 on tpu
      if self.args.runtime_tpu:
        # note: on GPUs we could use 'mixed_float32' but
        # then we would need to wrap the optimizers in LossScaleOptimizer! (not needed for bfloat32)
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

      print(f'Precision:')
      print(f'  Compute dtype:  {tf.keras.mixed_precision.experimental.global_policy().compute_dtype}')
      print(f'  Variable dtype: {tf.keras.mixed_precision.experimental.global_policy().variable_dtype}')
      print()
      print(f'Discriminator/Generator balance:')
      print(f'  {self.args.n_discr} discriminator updates for each generator update')
      print()

      # build keras model
      print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Building model...")
      self.model = model_factory.build_model(stage, self.global_batch_size, mode)

      # set up optimizers
      #   https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
      # SpecGAN lr=1e-4, beta_1=0.5, beta_2=0.9  <=== works best
      # ProGAN  lr=1e-3, beta_1=0.1, beta_2=0.99
      # BigGAN  lr_G=1e-4 lr_D=4e-4 (for batch<1024), beta_1=0.0, beta_2=0.999
      # https://cs231n.github.io/neural-networks-3/
      self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
      self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

      # set up checkpoint manager
      print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Setting up checkpoint manager...")
      self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                            discriminator_optimizer=self.discriminator_optimizer,
                                            generator=self.model.generator,
                                            discriminator=self.model.discriminator,
                                            song_counter=self.model.song_counter)  # what to save
      self.checkpoint_dir = gspath.join(args.training_dir, "stage-{}".format(self.model.stage))
      self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5,
                                                           checkpoint_name="stage-{}-ckpt".format(self.model.stage))
      self.checkpoint_freq = args.train_checkpoint_freq

      # set up summary
      self.summary_audio_repr = AudioRepresentation(self.model.sample_rate, self.model.freq_n, compute_dtype=tf.float32)
      self.summary_freq = args.summary_freq
      self.summary_dir = summary_dir
      self.summary_writer = None
      print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Summaries are written to {self.summary_dir}")

  def __del__(self):
    # make sure to close summary writer
    if self.summary_writer is not None:
      self.summary_writer.close()

  def load_latest_checkpoint(self):
    with self.strategy.scope():
      # fetch latest checkpoint if any (and only complain if we find stuff in tf-graph which is not in checkpoint)
      latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
      if latest_checkpoint:
        self.checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: Restored from {latest_checkpoint}")
      else:
        print(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}: No checkpoint found for stage {self.model.stage}: initializing from scratch in stage {self.model.stage}")
    return

  def load_from_model_weights(self, weights, name_mapping):
    with self.strategy.scope():
      print("Copying weights from previous stage {}".format(self.model.stage, self.model.stage-1))
      # copy weights from earlier model
      old_g_weights, old_d_weights = weights
      old_weights = {**old_g_weights, **old_d_weights}  # merge dicts

      for new_model_layer in self.model.generator.layers + self.model.discriminator.layers:
        if new_model_layer.get_weights():

          # translate new layer name to old layer name
          equivalent_old_model_layer_name = name_mapping(new_model_layer.name)

          # copy weights, if equivalent old layer
          if equivalent_old_model_layer_name in old_weights:
            print("  {0} {1} <-- old {2} {3}".format(new_model_layer.name, [tf.shape(g_weights).numpy() for g_weights in new_model_layer.get_weights()],
                                               equivalent_old_model_layer_name, [tf.shape(g_weights_old).numpy() for g_weights_old in old_weights[equivalent_old_model_layer_name]]))
            new_model_layer.set_weights(old_weights[equivalent_old_model_layer_name])
          else:
            print("  {0} {1} <-- new weights".format(new_model_layer.name, [tf.shape(g_weights).numpy() for g_weights in new_model_layer.get_weights()]))
    return

  def print_model_summary(self):
    with self.strategy.scope():
      print()
      print("Already completed iterations in stage {0}  = {1:11,}".format(self.model.stage,
                                                                          self.discriminator_optimizer.iterations.numpy()))
      print("Total iterations to complete stage {0}     = {1:11,}".format(self.model.stage,
                                                                          self.discriminator_optimizer.iterations + (
                                                                                    self.model.stage_total_songs - self.model.song_counter.numpy()) / self.global_batch_size))
      print("Remaining iterations to complete stage {0} = {1:11,.0f}".format(self.model.stage, (
                self.model.stage_total_songs - self.model.song_counter.numpy()) / self.global_batch_size))
      print()
      print(f"Total songs to listen to in stage {self.model.stage}      = {self.model.stage_total_songs:11,}")
      print(f"Songs already listened to in stage {self.model.stage}     = {self.model.song_counter.numpy():11,.0f}")
      print()

      self.model.summary(line_length=120)
      # self.model.generator.summary(line_length=180)
      # self.model.discriminator.summary(line_length=180)

  # ########### #
  # 1. Training #
  # ########### #
  def training_loop(self, input_filenames):
    """
    :param input_filenames:  file paths of pre-processed audio files
    :return:
    """
    with self.strategy.scope():
      # get the training data
      def dataset_fn(distributed_context):
        spectrogram_dataset = dataloader.load_dataset(
            distributed_context,
            input_filenames,
            global_batch_size=self.global_batch_size,
            slice_len=self.model.blocks_n,
            slice_hop=self.model.blocks_n,
            freq_n=self.model.freq_n,
            channels_n=self.model.channels_n,
            repeat=True,
            shuffle=True,
            shuffle_buffer_size=self.args.data_shuffle_buffer_size,
            tpu=self.args.runtime_tpu)
        return spectrogram_dataset

      # now distribute the dataset over devices
      # note: The tf.data.Dataset returned by dataset_fn should have a per_replica_batch_size
      #   (see https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy)
      # experimental_distribute_datasets_from_function() returns a DataSet of PerReplica objects ;-)
      spectrogram_dataset_distr = self.strategy.experimental_distribute_datasets_from_function(dataset_fn)

      # Main training loop runs on CPU!!!
      step = tf.cast(self.discriminator_optimizer.iterations, dtype=tf.int64)
      for reals_batch_per_replica in spectrogram_dataset_distr:
        # real_mdct_norm_batch is a PerReplica object with per replica a batch with the per_replica_batch_size

        if tf.math.floormod(step, self.summary_freq) == 0 and step > 0:
          # write summary (not at same time as training, since it blows up the TPU...)
          self.write_summary_gan(reals_batch_per_replica, step)
          step += 1  # so for next batch from dataset, it takes the train branch of this if statement
        else:
          # train
          step = self.train_step(reals_batch_per_replica)
          step = tf.cast(step, dtype=tf.int64)
          # song_counter is not a distributed variable, so update needs to happen on host (not on replicas!)
          self.model.song_counter.assign_add(self.global_batch_size)

        # save checkpoint
        if tf.math.floormod(step, self.checkpoint_freq) == 0 and step > 0:
          print(f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}: Step {step}, saving checkpoint', end='')
          checkpoint_save_path = self.checkpoint_manager.save()
          print(f' at {checkpoint_save_path}... done')

        # profile the TPU (needs to be run in Colab which has the TPU)
        if self.args.runtime_tpu and self.args.tpu_profiling and tf.math.floormod(step, 51) == 0:
          trace_duration_ms = 6 * 1000   # needs to be long enough to capture a few training steps
          trace_launch_s = 10
          print(f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}: Step {step}, launching trace in {trace_launch_s} seconds...')
          subprocess.Popen(f"sleep {trace_launch_s} && capture_tpu_profile --logdir={self.args.training_dir} --duration_ms={trace_duration_ms} --service_addr={os.environ['COLAB_TPU_ADDR']}", shell=True)

        # exit condition
        if tf.greater_equal(self.model.song_counter, self.model.stage_total_songs):
          break

    return self._weights_by_layer_name(self.model.generator), self._weights_by_layer_name(self.model.discriminator)

  @tf.function
  def train_step(self, reals):
    """Crucial that this is a separate function with @tf.function
       otherwise it won't run on a GPU. But not important for TPU
    """
    # launch the TPU monster
    # strategy.run() returns PerReplica objects (dict-like structures) either use:
    # 1. experimental_local_results to get result per worker in tuple, or
    # 2. self.strategy.reduce() to reduce over workers, or
    # 3. direct dict structure of PerReplica object (see self._merge_batch_over_replicas(...))
    step_per_replica = self.strategy.run(self.train_step_per_replica_gan, args=(reals,))

    # reduce step_per_replica over the different replicas
    step = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, step_per_replica, axis=None)
    return step

  # This function should NOT have a @tf.function, does not work on TPU
  def train_step_per_replica_gan(self, reals_with_stddev):
    reals = reals_with_stddev[:, :, :, :, 0]
    masking_threshold = reals_with_stddev[:, :, :, :, 1]
    real_noised = self.model.audio_representation.add_noise(reals, masking_threshold)

    per_replica_batch_size = reals.shape[0]

    # discriminator training
    z_vector = tf.random.normal(shape=(per_replica_batch_size, self.model.latent_dim))
    with tf.GradientTape() as tape:
      d_loss_ave = self.model.d_loss(z_vector, real_noised, training=True)
    discriminator_gradients = tape.gradient(d_loss_ave, self.model.discriminator.trainable_variables)
    # When apply_gradients is called within a distribution strategy scope, its behavior is modified.
    # Specifically, before applying gradients on each parallel instance during synchronous training,
    # it performs a sum-over-all-replicas of the gradients.
    # see: https://www.tensorflow.org/guide/distributed_training
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.model.discriminator.trainable_variables))

    step = tf.cast(self.discriminator_optimizer.iterations, tf.float32)

    # generator training
    # less often than discriminator: only a very good discr can get the generator to move in the right direction!
    if self.discriminator_optimizer.iterations % self.args.n_discr == 0:
      z_vector = tf.random.normal(shape=(per_replica_batch_size, self.model.latent_dim))
      with tf.GradientTape() as tape:
        g_loss_ave = self.model.g_loss(z_vector, training=True)
      generator_gradients = tape.gradient(g_loss_ave, self.model.generator.trainable_variables)
      self.generator_optimizer.apply_gradients(zip(generator_gradients, self.model.generator.trainable_variables))

    return step

  # ########### #
  # 2. Summary  #
  # ########### #
  def get_distributed_dataset(self, input_filenames):
    """
    :param input_filenames:  file paths of pre-processed audio files
    :return:
    """
    with self.strategy.scope():
      # get the training data
      def dataset_fn(distributed_context):
        spectrogram_dataset = dataloader.load_dataset(
            distributed_context,
            input_filenames,
            global_batch_size=self.global_batch_size,
            slice_len=self.model.blocks_n,
            slice_hop=self.model.blocks_n,
            freq_n=self.model.freq_n,
            channels_n=self.model.channels_n,
            repeat=True,
            shuffle=True,
            shuffle_buffer_size=self.args.data_shuffle_buffer_size,
            tpu=self.args.runtime_tpu)
        return spectrogram_dataset

      # now distribute the dataset over devices
      # note: The tf.data.Dataset returned by dataset_fn should have a per_replica_batch_size
      #   (see https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy)
      # experimental_distribute_datasets_from_function() returns a DataSet of PerReplica objects ;-)
      return iter(self.strategy.experimental_distribute_datasets_from_function(dataset_fn))

  def evaluation_loop(self, distributed_dataset_iterator):
    with self.strategy.scope():
      # Main training loop runs on CPU!!!
      reals_batch_per_replica = distributed_dataset_iterator.get_next()
      # real_mdct_norm_batch is a PerReplica object with per replica a batch with the per_replica_batch_size

      step = tf.cast(self.discriminator_optimizer.iterations, dtype=tf.int64)

      # write summary
      print(f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}: Writing summary for iteration {step.numpy()}...')
      step = self.write_summary_gan(reals_batch_per_replica, step)

    return step

  def write_summary_gan(self, reals_batch_per_replica, step):
    # make sure a summery_writer is open
    if self.summary_writer is not None and tf.math.floormod(step, 5 * self.summary_freq) == 0 and step > 0:
      self.summary_writer.close()
      self.summary_writer = None

    if self.summary_writer is None:
      self.summary_writer = SummaryWriter(self.summary_dir)
      print(f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}: Opened new summary file... ')

    print(f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")}: Step {step}, writing summary... ', end='')

    # compute 1 full batch on all replicas and track the metrics
    loss_tuple, metric_tuple, score_tuple, fakes_tuple = self.eval_step_gan(reals_batch_per_replica)

    # unpack
    g_loss, d_loss = loss_tuple
    wdistance, grad_penalty, drift = metric_tuple
    fake_scores, real_scores = score_tuple
    fakes_all, fakes_all_noised = fakes_tuple

    # make the reals here
    reals_with_stddev = self._merge_batch_over_replicas(reals_batch_per_replica)

    reals = reals_with_stddev[:, :, :, :, 0]
    masking_threshold = reals_with_stddev[:, :, :, :, 1]
    reals_noised = self.model.audio_representation.add_noise(reals, masking_threshold)

    # find best, worst and mid scores (best first, ie. higher scores first)
    fakes_sorted_by_index = tf.argsort(tf.squeeze(fake_scores), direction='DESCENDING')
    reals_sorted_by_index = tf.argsort(tf.squeeze(real_scores), direction='DESCENDING')
    n = fakes_sorted_by_index.shape[0]

    fakes = tf.gather(fakes_all, [fakes_sorted_by_index[0], fakes_sorted_by_index[n // 2], fakes_sorted_by_index[-1]], axis=0)
    reals = tf.gather(reals, [reals_sorted_by_index[0], reals_sorted_by_index[n // 2], reals_sorted_by_index[-1]], axis=0)
    reals_noise = tf.gather(reals_noised, [reals_sorted_by_index[0], reals_sorted_by_index[n // 2], reals_sorted_by_index[-1]], axis=0)

    np.set_printoptions(edgeitems=10, linewidth=500)
    print()
    print(f"Fakes[0, :, :, 0]:")
    print(fakes[0, :, :, 0])
    print(f"Minimum = {tf.reduce_min(tf.where(fakes == 0, 1., tf.abs(fakes)))}")

    fakes = tf.cast(fakes, dtype=tf.float32)
    reals_noise = tf.cast(reals_noise, dtype=tf.float32)

    # write scalars...
    self.summary_writer.add_scalar('10_W_distance', wdistance.numpy(), global_step=step)

    self.summary_writer.add_scalar('20_loss/D', d_loss.numpy(), global_step=step)
    self.summary_writer.add_scalar('20_loss/D_GP', grad_penalty.numpy(), global_step=step)
    self.summary_writer.add_scalar('20_loss/D_drift', drift.numpy(), global_step=step)
    self.summary_writer.add_scalar('20_loss/G', g_loss.numpy(), global_step=step)

    self.summary_writer.add_scalar("50_progression/Fade-in", tf.reshape(self.model.fade_in(), shape=[]).numpy(), global_step=step)
    self.summary_writer.add_scalar("50_progression/Drown", self.model.drown().numpy(), global_step=step)

    fake_tonality = self.summary_audio_repr.tonality(fakes)
    real_tonality = self.summary_audio_repr.tonality(reals_noise)
    self.summary_writer.add_scalars("70_tonality/fakes", {'0_ave': tf.reduce_mean(fake_tonality).numpy(),
                                              '1_best': tf.reduce_mean(fake_tonality[0:1, :, :, :]).numpy()}, global_step=step)
    self.summary_writer.add_scalar("70_tonality/reals", tf.reduce_mean(real_tonality).numpy(), global_step=step)

    # spectrogram
    for i in range(3):
      # fake spectrograms
      self.summary_writer.add_images(f'10_fake/{i}',
                         self.summary_audio_repr.repr_to_spectrogram(fakes)[i, :, :, :].numpy(), global_step=step, dataformats='HWC')
      self.summary_writer.add_images(f'11_fake_intensity/{i}',
                         self.summary_audio_repr.repr_to_spectrogram(fakes, intensity=True, cmap=cm.CMRmap)[i, :, :, :].numpy(), global_step=step, dataformats='HWC')

      # real spectrograms
      self.summary_writer.add_images(f'20_real_noise/{i}',
                         self.summary_audio_repr.repr_to_spectrogram(reals_noise)[i, :, :, :].numpy(), global_step=step, dataformats='HWC')
      self.summary_writer.add_images(f'21_real_noise_intensity/{i}',
                         self.summary_audio_repr.repr_to_spectrogram(reals_noise, intensity=True, cmap=cm.CMRmap)[i, :, :, :].numpy(), global_step=step, dataformats='HWC')

    # audio (only if model is in final stage)
    if self.model.freq_n == self.model.audio_representation.freq_n:
      infer_dir = self.args.infer_dir
      if infer_dir is not None and not gspath.is_dir(infer_dir):
        gspath.mkdir(infer_dir)

      wav_fake = self.summary_audio_repr.repr_to_audio(fakes)
      wav_real_noise = self.summary_audio_repr.repr_to_audio(reals_noise)
      for i in range(3):
        self.summary_writer.add_audio(f'1_fake/{i}', wav_fake[i, :, :].numpy(), global_step=step, sample_rate=self.model.sample_rate)
        self.summary_writer.add_audio(f'2_real_noise/{i}', wav_real_noise[i, :, :].numpy(), global_step=step, sample_rate=self.model.sample_rate)

        if infer_dir is not None:
          audio_utils.save_audio(gspath.join(infer_dir, f'fake_sample{i}.wav'), wav_fake[i, :, :].numpy(), sample_rate=self.model.sample_rate, out_format='wav')
          audio_utils.save_audio(gspath.join(infer_dir, f'real_noise_sample{i}.wav'), wav_real_noise[i, :, :].numpy(), sample_rate=self.model.sample_rate, out_format='wav')

    # histograms
    self.summary_writer.add_histogram('1_fake', fake_scores.numpy(), global_step=step)
    self.summary_writer.add_histogram('2_real', real_scores.numpy(), global_step=step)

    self.summary_writer.add_scalars(f"11_score", {'0_fake_min': tf.reduce_min(fake_scores).numpy(),
                                      '1_fake_max': tf.reduce_max(fake_scores).numpy(),
                                      '2_real_min': tf.reduce_min(real_scores).numpy(),
                                      '3_real_max': tf.reduce_max(real_scores).numpy()}, global_step=step)

    print(f'done')

    return step

  @tf.function
  def eval_step_gan(self, reals_batch_per_replica):
    # launch the TPU monster
    (loss_tuple_per_replica, metric_tuple_per_replica, score_tuple_per_replica,
     fakes_tuple_per_replica) = self.strategy.run(self.eval_step_per_replica_gan, args=(reals_batch_per_replica,))

    # unpack
    g_loss_per_replica, d_loss_per_replica = loss_tuple_per_replica
    wdistance_per_replica, grad_penalty_per_replica, drift_ave_per_replica  = metric_tuple_per_replica
    fake_scores_per_replica, real_scores_per_replica = score_tuple_per_replica
    per_replica_fakes, per_replica_fakes_noise = fakes_tuple_per_replica

    # **SUM** over each replica (is already normalized to get a mean -- just like the gradients in apply_gradients())
    g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_per_replica, axis=None)
    d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_per_replica, axis=None)

    wdistance = self.strategy.reduce(tf.distribute.ReduceOp.SUM, wdistance_per_replica, axis=None)
    grad_penalty = self.strategy.reduce(tf.distribute.ReduceOp.SUM, grad_penalty_per_replica, axis=None)
    drift = self.strategy.reduce(tf.distribute.ReduceOp.SUM, drift_ave_per_replica, axis=None)

    # re-package
    loss_tuple = (g_loss, d_loss)
    metric_tuple = (wdistance, grad_penalty, drift)
    score_tuple = (self._merge_batch_over_replicas(fake_scores_per_replica),
                   self._merge_batch_over_replicas(real_scores_per_replica))
    fakes_tuple = (self._merge_batch_over_replicas(per_replica_fakes),
                   self._merge_batch_over_replicas(per_replica_fakes_noise))

    return loss_tuple, metric_tuple, score_tuple, fakes_tuple

  # This function should NOT have a @tf.function, does not work on TPU
  def eval_step_per_replica_gan(self, reals_with_stddev):
    reals = reals_with_stddev[:, :, :, :, 0]
    masking_threshold = reals_with_stddev[:, :, :, :, 1]
    real_noised = self.model.audio_representation.add_noise(reals, masking_threshold)

    per_replica_batch_size = reals.shape[0]

    # compute
    z_vector = tf.random.normal(shape=(per_replica_batch_size, self.model.latent_dim))
    d_loss_ave, metric_tuple, score_tuple, fakes_tuple = self.model.d_loss_verbose(z_vector, real_noised, training=False)
    g_loss_ave = self.model.g_loss_verbose(z_vector, training=False)

    return (g_loss_ave, d_loss_ave), metric_tuple, score_tuple, fakes_tuple

  # ############ #
  # 3. Inference #
  # ############ #
  def inference_loop(self):
    inference_dir = gspath.join(self.args.training_dir, "infer")
    if not gspath.is_dir(inference_dir):
      gspath.mkdir(inference_dir)

    with self.strategy.scope():
      _, l1, l2, l3, l4, _ = self.infer_step()

      timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

      imageio.imwrite(os.path.join(inference_dir, 'l{0}_64x16_deep_strip_mean_of_squares_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(tf.sqrt(tf.reduce_mean(l1 ** 2, axis=-1, keepdims=True))),
                                                                          intensity=True)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_64x16_deep_strip_ave_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(tf.reduce_mean(l1, axis=-1, keepdims=True)),
                                                                          intensity=True)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_64x16_deep_strip_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(l1),
                                                                          intensity=True)[0, :, :])

      imageio.imwrite(os.path.join(inference_dir, 'l{0}_256x32_deep_strip_ave_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(tf.reduce_mean(l2, axis=-1, keepdims=True)),
                                                                          intensity=True)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_256x32_deep_strip_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(l2),
                                                                          intensity=True)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_256x32_blurred_output_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self.model.blur_layer(l4),
                                                                          intensity=True)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_256x32_blurred_output.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self.model.blur_layer(l4),
                                                                          intensity=False)[0, :, :])


      imageio.imwrite(os.path.join(inference_dir, 'l{0}_1024x64_output_strip_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(self._collate_probe_tensors(l3),
                                                                          intensity=True)[0, :, :])

      imageio.imwrite(os.path.join(inference_dir, 'l{0}_1024x64_output.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(l4,
                                                                          intensity=False)[0, :, :])
      imageio.imwrite(os.path.join(inference_dir, 'l{0}_1024x64_output_intensity.png'.format(timestamp)),
                      self.model.audio_representation.repr_to_spectrogram(l4,
                                                                          intensity=True)[0, :, :])

      wave_data = self.model.audio_representation.repr_to_audio(l4)[0, :, :]

      audio_utils.save_audio(os.path.join(inference_dir, 'l{0}_audio.wav'.format(timestamp)),
                             wave_data, self.model.audio_representation.sample_rate)

      # tensors_to_plot = audio_util_lab.reduce_spectrogram(l4, self.model)
      # audio_util_lab.plot(tensors_to_plot)

  def _collate_probe_tensors(self, probe_tensor):
    print(f"tensor.shape = {probe_tensor.shape}; max = {tf.reduce_max(tf.abs(probe_tensor))}")
    probe_tensor = probe_tensor / tf.reduce_max(tf.abs(probe_tensor))
    padded = tf.pad(probe_tensor, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), mode='CONSTANT', constant_values=.5)
    transposed = tf.transpose(padded, perm=[0, 3, 1, 2])
    probe = tf.expand_dims(tf.reshape(transposed, shape=[tf.shape(transposed)[0], -1, tf.shape(transposed)[3]]), axis=-1)
    return probe

  @tf.function
  def infer_step(self):
    # only written for inference on GPU, not TPU
    fakes_and_probes_per_replica = self.strategy.run(self.infer_step_per_replica)

    # no self._merge_batch_over_replicas() necessary for TPU is applied
    return fakes_and_probes_per_replica

  # This function should NOT have a @tf.function, does not work on TPU
  def infer_step_per_replica(self):
    number_of_fakes = 3

    # create fakes
    z_vector = tf.random.truncated_normal(shape=(number_of_fakes, self.model.latent_dim))
    fakes_and_probes = self.model.generator(z_vector, training=False)
    fakes_masking_threshold = self.model.audio_representation.psychoacoustic_masking_ampl(fakes_and_probes)
    fakes_and_probes_noised = self.model.audio_representation.add_noise(fakes_and_probes, fakes_masking_threshold)

    return fakes_and_probes_noised

  def _merge_batch_over_replicas(self, per_replica_tensor):
    if self.strategy.num_replicas_in_sync > 1:
      # per_replica_tensor is a PerReplica object (dict-like) of tensors (one tensor per replica),
      # so we stack them along the 0th dimension (batch) of the tensors itself
      if len(per_replica_tensor.values[0].shape.as_list()) > 1:
        shape = [-1] + per_replica_tensor.values[0].shape.as_list()[1:]
      else:
        shape = [-1]
      tensor = tf.reshape(tf.stack(per_replica_tensor.values, axis=0), shape=shape)
      # note can also be implemented with tf.distribute.Strategy.experimental_local_results()
    else:
      tensor = per_replica_tensor
    return tensor

  def _weights_by_layer_name(self, model):
    return {layer.name: layer.get_weights() for layer in model.layers if layer.get_weights()}
