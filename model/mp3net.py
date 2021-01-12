# !/usr/bin/env python

"""Definition of MP3net model"""

import numpy as np
import tensorflow as tf

from model.audiorepresentation import AudioRepresentation

import utils

import layers


# Model structure
#
# 95s MAESTRO (6 blocks)
# ======================
#
#  22016Hz  x 95.3s
# 128 x 172 x 95.3s
# 128   x 16384        ==> full image size (2 more blocks & 512 channel depth)
#  64   x  4096
#  32   x  1024
#  16   x   256
#   8   x    64
#   4   x    16
#   1   x     4   <=== base


_EPS = 1e-8


class MP3netFactory:
  def __init__(self):
    # initialization of audio & spectrum transformations
    # ProGAN training phases
    self.stages = 6

    self.stage_total_songs = [1024 * x for x in [0] * (self.stages-1) + [200000]]
    self.fade_in_progression = [
      lambda song_counter, stage=stage: tf.constant(1., shape=())
      for stage in range(self.stages)]
    self.drown_progression = [
      lambda song_counter: tf.constant(3., shape=())
      for _ in range(self.stages)]

    # CNN parameters
    self.base_block_length = 4
    self.dim_ch = 2*8
    self.max_dim_ch = 4*128
    self.blocks_per_sec = 172

    self.blocks_n = [4**(stage+1) * self.base_block_length for stage in range(self.stages)]
    self.freq_n = [2 * 2**(stage+1) for stage in range(self.stages)]  # \Sum_i^N 2^i = 2^N+1 - 1
    self.channels_n = [2 for _ in range(self.stages)]  # 1:mono 2:stereo

    self.drown = [0.0 for _ in range(self.stages)]  # sets amount of noise added to deeper samples

    self.sample_rate = self.freq_n[-1] * self.blocks_per_sec
    self.latent_dim, _, _ = self.generator_filters_n(block_i=0)

  @staticmethod
  def map_layer_names_for_model_growth(new_model_layer_name):
    equivalent_old_model_layer_name = new_model_layer_name

    if equivalent_old_model_layer_name == 'generator_output_conv2d_old':
      equivalent_old_model_layer_name = 'generator_output_conv2d_new'
    elif equivalent_old_model_layer_name == 'generator_output_conv2d_new':
      equivalent_old_model_layer_name = ''
    if equivalent_old_model_layer_name == 'discriminator_output_conv2d_old':
      equivalent_old_model_layer_name = 'discriminator_output_conv2d_new'
    elif equivalent_old_model_layer_name == 'discriminator_output_conv2d_new':
      equivalent_old_model_layer_name = ''

    return equivalent_old_model_layer_name

  def generator_block(self, x, generator_block_i):
    in_filters_n, mid_filters_n, out_filters_n = self.generator_filters_n(generator_block_i)

    # philosophy: drop channels, when we are upsampling!
    block_kernel_length = 3
    freq_kernel_length = 3

    # Make new high octave: (1, 2) up-scaling + Conv2D + Conv2D
    freq_n = x.shape[2]
    octaves_n = int(np.log(freq_n + 1) / np.log(2.))
    lower = x
    higher = x[:, :, 2 ** (octaves_n - 1):, :]

    # Conv2DTranspose = up-sampling(2, 2) + Conv2D(1, 1)
    higher = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                             strides=(2, 2), padding='same', name='generator_block_{}.0.2x2H'.format(generator_block_i))(higher)
    higher = tf.keras.layers.LeakyReLU(alpha=0.2)(higher)

    # Conv2DTranspose = Conv2D(1, 1)
    higher = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                             strides=(1, 1), padding='same', name='generator_block_{}.1.1x1H'.format(generator_block_i))(higher)
    higher = tf.keras.layers.LeakyReLU(alpha=0.2)(higher)

    # Conv2DTranspose = up-sampling(2, 1) + Conv2D(1, 1)
    lower = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                            strides=(2, 1), padding='same', name='generator_block_{}.0.2x1L'.format(generator_block_i))(lower)
    lower = tf.keras.layers.LeakyReLU(alpha=0.2)(lower)

    # Conv2DTranspose = Conv2D(1, 1)
    lower = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                            strides=(1, 1), padding='same', name='generator_block_{}.1.1x1L'.format(generator_block_i))(lower)
    lower = tf.keras.layers.LeakyReLU(alpha=0.2)(lower)

    # LeakyReLU assumes value distribution is centered around the discontinuity at 0
    # here we rebalance lower & higher relative to each other before gluing back together (NO BatchNorm here!!)
    higher = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(1, 1),
                                             strides=(1, 1), padding='same', name=f'generator_block_{generator_block_i}.2.1x1H')(higher)
    lower = tf.keras.layers.Conv2DTranspose(filters=mid_filters_n, kernel_size=(1, 1),
                                            strides=(1, 1), padding='same', name=f'generator_block_{generator_block_i}.2.1x1L')(lower)
    x = tf.keras.layers.Concatenate(axis=2)([lower, higher])

    # Conv2DTranspose = up-sampling(2, 1) + Conv2D(1, 1)
    x = tf.keras.layers.Conv2DTranspose(filters=out_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                        strides=(2, 1), padding='same', name='generator_block_{}.3.2x1'.format(generator_block_i))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Conv2DTranspose = Conv2D(1, 1)
    x = tf.keras.layers.Conv2DTranspose(filters=out_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                        strides=(1, 1), padding='same', name='generator_block_{}.4.1x1'.format(generator_block_i))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x

  def discriminator_block(self, x, discriminator_block_i, shaving=False):
    # Note: no BatchNormalization since we use GP on the discriminator
    in_filters_n, mid_filters_n, out_filters_n = self.discriminator_filters_n(discriminator_block_i)

    # 1. Full spectrum: Conv2D + Conv2D + Down-sample 2x(2, 1)
    block_kernel_length = 3  # *2
    freq_kernel_length = 3  # *4

    x = tf.keras.layers.Conv2D(filters=in_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                               strides=(1, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.4.1x1')(x)
    if shaving: x = x[:, 1:, :, :]
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                               strides=(2, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.3.2x1')(x)
    if shaving: x = x[:, :-1, :, :]
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # inversion of the concat layer in the generator
    freq_n = x.shape[2]
    octaves_n = int(np.log(freq_n + 1) / np.log(2.))
    lower = x[:, :, :2 ** (octaves_n - 1), :]
    higher = x[:, :, 2 ** (octaves_n - 1):, :]

    higher = tf.keras.layers.Conv2D(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                    strides=(1, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.2.1x1H')(higher)
    if shaving: higher = higher[:, 1:, :, :]
    higher = tf.keras.layers.LeakyReLU(alpha=0.2)(higher)

    higher = tf.keras.layers.Conv2D(filters=out_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                    strides=(2, 2), padding='same', name=f'discriminator_block_{discriminator_block_i}.1.2x2H')(higher)
    if shaving: higher = higher[:, :-1, :, :]
    higher = tf.keras.layers.LeakyReLU(alpha=0.2)(higher)

    lower = tf.keras.layers.Conv2D(filters=mid_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                   strides=(1, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.2.1x1L')(lower)
    if shaving: lower = lower[:, 1:, :, :]
    lower = tf.keras.layers.LeakyReLU(alpha=0.2)(lower)

    lower = tf.keras.layers.Conv2D(filters=out_filters_n, kernel_size=(block_kernel_length, freq_kernel_length),
                                   strides=(2, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.1.2x1L')(lower)
    if shaving: lower = lower[:, :-1, :, :]
    lower = tf.keras.layers.LeakyReLU(alpha=0.2)(lower)

    # rescale and shift before summing
    higher = tf.keras.layers.Conv2D(filters=out_filters_n, kernel_size=(1, 1),
                                    strides=(1, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.0.1x1H')(higher)
    lower = tf.keras.layers.Conv2D(filters=out_filters_n, kernel_size=(1, 1),
                                   strides=(1, 1), padding='same', name=f'discriminator_block_{discriminator_block_i}.0.1x1L')(lower)
    if shaving:
      lower = lower[:, 1:-1, :, :]
      higher = higher[:, 1:-1, :, :]
    higher = tf.pad(higher, tf.constant([[0, 0], [0, 0], [2 ** (octaves_n - 2), 0], [0, 0]]))   # pad lower octaves
    x = lower + higher

    return x

  def generator_filters_n(self, block_i):
    channels_in = 4 * 4 ** (self.stages - 1 - block_i) * self.dim_ch
    channels_mid = 2 * 4 ** (self.stages - 1 - block_i) * self.dim_ch
    channels_out = 1 * 4 ** (self.stages - 1 - block_i) * self.dim_ch

    def ceiling(x):
      return min(x, self.max_dim_ch)

    return ceiling(channels_in), ceiling(channels_mid), ceiling(channels_out)

  def discriminator_filters_n(self, block_i):
    channels_in, channels_mid, channels_out = self.generator_filters_n(block_i)
    return channels_out, channels_mid, channels_in

  def split_to_batch(self, x, split, padding):
    # pad the input
    x_padded = tf.pad(x, tf.constant([[0, 0], padding, [0, 0], [0, 0]]))
    # sliding window over width dimension
    split_width = x.shape[1] // split
    x_batched = tf.stack([x_padded[:, n * split_width:(n + 1) * split_width + sum(padding), :, :] for n in range(split)])
    # mix width with batch dimension
    x_batched = tf.transpose(x_batched, perm=[1, 0, 2, 3, 4])
    x_split = tf.reshape(x_batched, shape=(-1, split_width + sum(padding), x_padded.shape[2], x_padded.shape[3]))
    return x_split

  def join_from_batch(self, x, shave, join_width):
    # shave off extra edges
    x_shaved = x[:, shave[0]:x.shape[1] - shave[1], :, :]
    # move batch back to spatial
    x_joined = tf.reshape(x_shaved, shape=(-1, join_width, x.shape[2], x.shape[3]))
    return x_joined

  def make_generator_model(self, model_stage, global_batch_size, with_probe=False):
    """generator output is -1 < ... < 1 (tanh activation!)
    [batch_size,
       [blocks_n, freqs_n, in_channels]
    ]
    :param model_stage:  progressive stage of model training
    :param with_probe:   not a Tensor, but a python variable to force re-tracing (so we get a faster and slower Graph)
    :return:             the keras generator model
    """
    per_worker_batch_size = global_batch_size // tf.distribute.get_strategy().num_replicas_in_sync

    inputs = tf.keras.Input(batch_input_shape=(per_worker_batch_size, self.latent_dim), name='generator_latent_input')
    probes = []

    _, _, out_filters_n = self.generator_filters_n(block_i=0)

    # FC (= Conv2DTranspose(kernel=Xx3, 'VALID')
    x = tf.keras.layers.Dense(self.base_block_length * 2 * self.latent_dim, name=f'generator_conv2dt.{self.base_block_length}x1')(inputs)
    x = tf.keras.layers.Reshape((self.base_block_length, 2, self.latent_dim))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    block_i_range = range(model_stage + 1)
    for generator_block_i in block_i_range[:-2]:
      x = self.generator_block(x, generator_block_i)
      if with_probe:
        probes.append(x)

    x = self.split_to_batch(x, split=128, padding=[4, 2])

    generator_block_i = model_stage - 1
    x = self.generator_block(x, generator_block_i)
    generator_block_i = model_stage
    x = self.generator_block(x, generator_block_i)

    # note: bias is one single constant up-lifting the full output; initially it needs to pass through the pa-filter!
    x = tf.keras.layers.Conv2D(filters=self.channels_n[model_stage], kernel_size=(1, 1), strides=(1, 1), padding='same',
                               name='generator_output_conv2d_new', use_bias=True)(x)
    # no Leaky ReLU or BatchNorm, as this is the actual output! (similar to ProGAN)
    x = self.join_from_batch(x, shave=[4*16, 2*16], join_width=self.blocks_n[model_stage])

    # not present in ProGAN
    # we need it, since psychoacoustic filter expects input in -1..1 range
    # earlier we used clipping, but derivatives of clip result in zero gradients hence no learning(?)
    x = tf.cast(x, dtype=tf.float32)   # does this fix the strangely quantized output of the generator...?
    x = tf.tanh(x)

    if with_probe:
      probes.extend([x])
      model = tf.keras.Model(inputs=inputs, outputs=probes, name='generator')
    else:
      model = tf.keras.Model(inputs=inputs, outputs=x, name='generator')

    return model

  def make_discriminator_model(self, model_stage, global_batch_size, mode):
    """
    Output of the discriminator is a logit (-inf..inf)
    No batch normalization when using gradient penalty!!

    :param model_stage:  phase of model training
    :return:             critic output, [1] (= scalar)
    """
    per_worker_batch_size = global_batch_size // tf.distribute.get_strategy().num_replicas_in_sync
    batch_input_shape = tf.TensorShape([per_worker_batch_size, self.blocks_n[model_stage],
                                        self.freq_n[model_stage], self.channels_n[model_stage]])
    inputs = tf.keras.Input(batch_input_shape=batch_input_shape, name='discriminator_input')

    # let keras autocast-magic convert inputs to bfloat16 when in mixed_precision mode
    x = tf.keras.layers.Activation('linear')(inputs)

    x = self.split_to_batch(x, split=128, padding=[35, 50])

    # first discriminator block:
    # split input into channels
    discriminator_block_i = model_stage
    in_filter_n, _, out_filter_n = self.discriminator_filters_n(discriminator_block_i)
    x = tf.keras.layers.Conv2D(filters=in_filter_n, kernel_size=(1, 1),
                               strides=(1, 1), padding='same', name='discriminator_output_conv2d_new')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = self.discriminator_block(x, discriminator_block_i, shaving=True)

    discriminator_block_i = model_stage - 1
    x = self.discriminator_block(x, discriminator_block_i, shaving=True)

    x = self.join_from_batch(x, shave=[0, 0], join_width=self.blocks_n[model_stage] // (4 * 4))

    for discriminator_block_i in list(reversed(range(model_stage + 1)))[2:]:
      x = self.discriminator_block(x, discriminator_block_i)

    # add one feature checking the std dev of pixels
    x = layers.BatchStddevLayer(batch_group_size=1, device_group_size=8)(x)

    # Dense [batches_n, 4, 4-1, 8*dim] -> [batches_n, 8*dim]
    # One can also see this as a final convolution layer
    _, _, out_filters_n = self.discriminator_filters_n(block_i=0)

    # Conv2D(kernel=3x3, 'VALID')
    # [batches_n, self.base_block_length, 3, channels_n] --> [batches_n, 1, 1, channels_n]
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=out_filters_n, name=f'discriminator_conv2d.{self.base_block_length}x1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Connect all channels to single logit
    # [batches_n, 1, 1, channels_n] --> [batches_n, 1]
    # make sure output is float32 for numerical stability in gradients(?)
    #   (see https://www.tensorflow.org/guide/mixed_precision#building_the_model)
    x = tf.keras.layers.Dense(units=1, name='discriminator_fc', dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='discriminator')

    return model

  def build_model(self, stage, global_batch_size, mode):
    with_probe = mode.lower() == 'infer'
    # not yet initialized, since tf.distribute.Strategy context not yet defined
    assert tf.distribute.has_strategy(), "TranceModel needs to have a strategy context active"

    stage_total_songs = self.stage_total_songs[stage]
    fade_in_progression = self.fade_in_progression[stage]
    drown_progression = self.drown_progression[stage]

    blocks_n = self.blocks_n[stage]
    freq_n = self.freq_n[stage]
    channels_n = self.channels_n[stage]

    # generator = self.build_generator(stage)
    generator = self.make_generator_model(stage, global_batch_size, with_probe=with_probe)
    discriminator = self.make_discriminator_model(stage, global_batch_size, mode)

    return MP3net(stage, stage_total_songs, fade_in_progression, drown_progression,
                  blocks_n, freq_n, channels_n, self.sample_rate, self.latent_dim,
                  generator, discriminator,
                  global_batch_size)


class MP3net:
  def __init__(self, stage, stage_total_songs, fade_in_progression, drown_progression,
               blocks_n, freq_n, channels_n, sample_rate, latent_dim,
               generator: tf.keras.Model, discriminator: tf.keras.Model,
               global_batch_size):
    self.stage = stage
    self.stage_total_songs = stage_total_songs

    self.blocks_n = blocks_n
    self.freq_n = freq_n
    self.sample_rate = sample_rate
    self.channels_n = channels_n

    self.latent_dim = latent_dim

    self.audio_representation = AudioRepresentation(self.sample_rate, self.freq_n,
                                                    compute_dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)

    self.global_batch_size = tf.constant(global_batch_size, dtype=tf.float32)

    # build the models
    self.generator = generator
    self.discriminator = discriminator

    # progression functions
    self.fade_in_progression = fade_in_progression
    self.drown_progression = drown_progression

    # a tf.Variable so we can save it in checkpoint
    self.song_counter = tf.Variable(0, trainable=False, dtype=tf.float32)

  def fade_in(self):
    return self.fade_in_progression(self.song_counter)

  def drown(self):
    return self.drown_progression(self.song_counter)

  def summary(self, line_length=None):
    utils.tf_utils.summary(self.generator, line_length)
    utils.tf_utils.summary(self.discriminator, line_length)

  @tf.function
  def g_loss_verbose(self, z_vector, training: bool):
    # create fakes
    fakes_tf32 = self.generator(z_vector, training=training)
    fakes = tf.cast(fakes_tf32, dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)
    fakes_masking_threshold = self.audio_representation.psychoacoustic_masking_ampl(fakes)
    fakes_noised = self.audio_representation.add_noise(fakes, fakes_masking_threshold)

    # compute discriminator scores
    score_fake_ave = self._global_batch_mean(self.discriminator(fakes_noised, training=training))

    # minimizing g_loss will try to push discr_score_fake to +inf! (with fake = -inf, real = +inf)
    g_loss_ave = -score_fake_ave

    return g_loss_ave

  @tf.function
  def g_loss(self, z_vector, training: bool):
    g_loss = self.g_loss_verbose(z_vector, training)
    return g_loss

  @tf.function
  def d_loss_verbose(self, z_vector, reals_noised, training: bool):
    # create fakes
    fakes_tf32 = self.generator(z_vector, training=training)
    fakes = tf.cast(fakes_tf32, dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)
    fakes_masking_threshold = self.audio_representation.psychoacoustic_masking_ampl(fakes)
    fakes_noised = self.audio_representation.add_noise(fakes, fakes_masking_threshold)

    # compute discriminator scores
    discr_score_fake = self.discriminator(fakes_noised, training=training)
    discr_score_real = self.discriminator(reals_noised, training=training)

    # Wasserstein-1 = Earth Mover distance (EMD) between 2 distribution
    #   (see https://en.wikipedia.org/wiki/Wasserstein_metric)
    # Wasserstein-1 is not computed directly, rather in its dual Kantorovich-Rubinstein formulation,
    # which requires the function f (discriminator) to be Lipschitz-1 (so, we need a gradient penalty)
    #   (see section 3 of https://arxiv.org/pdf/1701.07875.pdf)
    wasserstein_distance = self._global_batch_mean(discr_score_fake - discr_score_real)

    # We could brute-force compute Wasserstein-1 = EMD directly on large batches with EMD algo
    # and drop the Lipschitz-1 constraint (so, no gradient penalty)
    # BUT, requiring the discriminator to be Lipschitz-1, makes our loss function
    # continuous and differentiable which improves the minimization
    #   (see theorem 1 of https://arxiv.org/pdf/1701.07875.pdf)
    grad_penalty_ave = self._global_batch_mean(
      self._gradient_penalty(reals_noised, fakes_noised))

    # tf.square term locks in place the real discriminator scores so they don't run away and overpower the grad_penalty
    #   (see appendix A.1 of https://arxiv.org/pdf/1710.10196.pdf)
    drift_ave = self._global_batch_mean(tf.square(discr_score_real))

    # high GP coefficient prevents model collapse, but reduces model performance (see BigGAN paper)
    d_loss_ave = wasserstein_distance + self.drown() * grad_penalty_ave + 0.1 * drift_ave

    return d_loss_ave, (wasserstein_distance, grad_penalty_ave, drift_ave), (discr_score_fake, discr_score_real), (fakes_tf32, fakes_noised)

  @tf.function
  def d_loss(self, z_vector, real_noised, training: bool):
    d_loss, _, _, _ = self.d_loss_verbose(z_vector, real_noised, training)
    return d_loss

  @tf.function
  def _gradient_penalty(self, reals, fakes):
    """

    :param reals:               a real input
    :param fakes:               a faked input (from generator)
    :return:                    penalty factor                        [batches_n, 1]
    """
    # instead of clipping the weights, this alternative condition also imposes the Lipschitz constraint
    # (see: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490 and
    # https://github.com/KUASWoodyLIN/TF2-WGAN/blob/master/utils/losses.py)
    # prediction should go from real (+inf) to fake (-inf)
    # pick an interpolation (alpha) and compute gradient there

    # different alpha for each element in batch
    batches_n = tf.shape(reals)[0]
    shape = [batches_n] + [1] * (reals.shape.ndims - 1)
    alpha = tf.random.uniform(shape, minval=0., maxval=1., dtype=reals.dtype)

    interpolated = alpha * reals + (1. - alpha) * fakes

    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      discr_score_interpolated = self.discriminator(interpolated, training=True)

    # with mixed_bfloat16, these gradients will be bfloat16
    gradients_new = tape.gradient(discr_score_interpolated, interpolated)

    # norm.shape = [batches_n, 1]
    norm = tf.sqrt(tf.reduce_sum(tf.reshape(gradients_new, shape=[tf.shape(gradients_new)[0], -1]) ** 2,
                                 axis=1, keepdims=True) + _EPS)

    # gradient penalty loss function (forces it to 1)
    # [batches_n, 1]
    # cast to float32 since entire loss function is float32
    return (tf.cast(norm, dtype=tf.float32) - 1.) ** 2.

  def _global_batch_mean(self, batch_from_replica):
    """
    Variables on replicas need to be divided by the global_batch_size (not the per_replica_batch_size)
    since apply_gradients() adds gradients of each replica (without dividing by the number of replica!)

    see: https://www.tensorflow.org/guide/distributed_training#examples_and_tutorials
      When apply_gradients is called within a distribution strategy scope, its behavior is modified.
      Specifically, before applying gradients on each parallel instance during synchronous training,
      it performs a sum-over-all-replicas of the gradients

    :param batch_from_replica:   batch of values from one replica to be averaged over all batches
    :return:                     average
    """
    return tf.reduce_sum(batch_from_replica) / tf.cast(self.global_batch_size, dtype=batch_from_replica.dtype)
