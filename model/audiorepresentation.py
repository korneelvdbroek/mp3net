# !/usr/bin/env python

"""Utility function for audio signal processing in specGAN"""

import numpy as np
import matplotlib.cm

import tensorflow as tf

from audiocodec.mdctransformer import MDCTransformer
from audiocodec.psychoacoustic import PsychoacousticModel


class AudioRepresentation:
  def __init__(self, sample_rate, freq_n, compute_dtype):
    # [batches_n, blocks_n, freqs_n, channels_n]
    # sample_rate = blocks_n x freqs_n

    self.sample_rate = tf.constant(sample_rate)
    self.freq_n = tf.constant(freq_n)

    # initialize MDCTransformer and PsychoacousticModel
    self.mdctransformer = MDCTransformer(freq_n, window_type='vorbis',
                                         compute_dtype=compute_dtype)
    # Note: Number of bark determines level of masking threshold!!! For freq_n = 256, bark_bands_n=24 is appropriate
    self.psychoacoustic = PsychoacousticModel(sample_rate, freq_n, bark_bands_n=24, alpha=0.6,
                                              compute_dtype=compute_dtype)

  @tf.function
  def t_to_repr(self, wave):
    """Convert audio signal to representation for neural model

    :param wave:         audio signal                    [batches_n, samples_n, channels_n]
    :return              audio representation            [batches_n, blocks_n+1, freqs_n, channels_n]
                         with samples_n = blocks_n * freq_n
    """
    samples_n = tf.shape(wave)[1]
    tf.assert_equal(tf.truncatemod(samples_n, self.freq_n), 0,
                    f'Number of samples ({samples_n}) needs to be a multiple of {self.freq_n}')

    return self.mdctransformer.transform(wave)

  @tf.function
  def repr_to_t(self, mdct_norm):
    """Convert representation to audio signal

    :param mdct_norm:    audio representation            [batches_n, blocks_n, freqs_n, channels_n]
    :return              audio signal                    [batches_n, samples_n, channels_n]
                         with samples_n = (blocks_n+1) * freq_n
    """
    return self.mdctransformer.inverse_transform(mdct_norm)

  @tf.function
  def tonality(self, mdct_norm):
    """Computes the tonality of the audio signal defined by the representation

    :param mdct_norm:           audio representation            [batches_n, blocks_n, freqs_n, channels_n]
    :return:                    tonality per block              [batches_n, blocks_n, 1, channels_n]
    """
    return self.psychoacoustic.tonality(mdct_norm)

  @tf.function
  def psychoacoustic_masking_ampl(self, mdct_norm, drown=0.0):
    """Get hearing threshold for each pixel in the spectrogram

    :param mdct_norm:           normalized mdct amplitudes      [batches_n, blocks_n, freqs_n, channels_n]
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :return:                    masking amplitude (positive)    [batches_n, blocks_n, freqs_n, channels_n]
    """
    tonality_per_block = self.psychoacoustic.tonality(mdct_norm)
    total_threshold = self.psychoacoustic.global_masking_threshold(mdct_norm, tonality_per_block, drown)
    return total_threshold

  @tf.function
  def add_noise(self, mdct_norm, masking_threshold):
    """
    Adds inaudible noise to amplitudes, using the masking_threshold.
    The noise added is calibrated at a 3-sigma deviation in both directions:
      masking_threshold = 6*sigma
    As such, there is a 0.2% probability that the noise added is bigger than the masking_threshold

    :param mdct_norm:           mdct amplitudes (spectrum) for each filter [batches_n, blocks_n, filter_bands_n, channels_n]
                                must be of compute_dtype
    :param masking_threshold:   masking threshold in amplitude. Masking threshold is never negative
                                output dtype is compute_dtype
                                [batches_n, blocks_n, filter_bands_n, channels_n]
    :return:                    mdct amplitudes with inaudible noise added [batches_n, blocks_n, filter_bands_n, channels_n]
    """
    return self.psychoacoustic.add_noise(mdct_norm, masking_threshold)

  @tf.function
  def psychoacoustic_filter(self, mdct_norm, masking_threshold, max_gradient=10):
    """Apply lRElu filter to tab-representation

    :param mdct_norm:           normalized mdct amplitudes      [batches_n, blocks_n, freqs_n, channels_n=1]
    :param masking_threshold:   masking threshold in amplitude. Masking threshold is never negative
                                output dtype is compute_dtype
                                [batches_n, blocks_n, filter_bands_n, channels_n]
    :param drown:               factor 0..1 to drown out audible sounds (0: no drowning, 1: fully drowned)
    :param max_gradient:        maximum gradient filter will introduce
    :return:                    normalized mdct amplitudes      [batches_n, blocks_n, freqs_n, channels_n=1]
    """
    # ReLU-filter
    def f_attentuation(x):
      # function with
      #   f(0)  = 0.
      #   f(1)  = 1.
      #   f'(0) = max_gradient / (2**(max_gradient+1) - 2)  >~  0
      #   f'(1) = max_gradient / (1. - 1./2**max_gradient)  >~  max_gradient
      return (1. / (2. - x)**max_gradient - 1./2**max_gradient) / (1. - 1./2**max_gradient)
    x_abs = tf.abs(mdct_norm) / masking_threshold
    x_abs_safe = tf.where(x_abs < 1., x_abs, 1.)
    mdct_norm_filtered = tf.where(x_abs < 1., f_attentuation(x_abs_safe) * mdct_norm, mdct_norm)

    return mdct_norm_filtered

  def repr_to_spectrogram(self, mdct_norm, intensity=False, channel=0, cmap=None):
    """Make image of normalized mdct amplitudes

    :param mdct_norm:           mdct amplitudes                   [batches_n, blocks_n, freqs_n, channels_n]
    :param intensity:           shows amplitudes if False, intensities if True
    :param channel:             select (stereo)-channel which needs to be displayed
    :param cmap:                matplotlib colormap
    :return:                    uint8 image with filter_band_n as height and #blocks as width
                                shape = [batches_n, blocks_n, freqs_n, color_channels]
                                where color_channels is 1 if cmap = None, otherwise it is 3 (RGB)
    """
    x = tf.cast(mdct_norm[:, :, :, channel:channel+1], tf.float32)

    def normalized_dB_scale(ampl, with_sign=True):
      normalized_dB = self.psychoacoustic.amplitude_to_dB_norm(ampl)
      if with_sign:
        # range -1..1
        return tf.sign(ampl) * normalized_dB
      else:
        # range 0..1
        return normalized_dB

    # convert to 0..1 range
    if intensity:
      image = normalized_dB_scale(x, with_sign=False)
    else:
      image = (normalized_dB_scale(x, with_sign=True) + 1.) / 2.

    image = tf.map_fn(lambda im: tf.image.rot90(im), image)

    # colorize with cmap
    if cmap is not None:
      # quantize
      image = image[:, :, :, 0]   # remove the dummy channel direction (will be replace with rgb info from color map)
      image_index = tf.cast(tf.round(image * (cmap.N-1)), dtype=tf.int32)   # indices in [0, cmap.N-1]

      image_index = tf.clip_by_value(image_index, clip_value_min=0, clip_value_max=cmap.N-1)

      # gather
      color_map = matplotlib.cm.get_cmap(cmap)(np.arange(cmap.N))   # shape=[cmap.N, 3]
      colors = tf.constant(color_map, dtype=tf.float32)
      image = tf.gather(colors, image_index)   # image[b, h, w, c] = color[image_index[b, h, w], c]

    return image

  def repr_to_audio(self, mdct_norm):
    """Make audio of mdct amplitudes

    :param mdct_norm:             mdct amplitudes                 [batches_n, blocks_n, freqs_n, channels_n]
    :return:                      audio signal                    [batches_n, samples_n, channels_n]
                                  with samples_n = (blocks_n+1) * freq_n
    """
    mdct_norm_ft32 = tf.cast(mdct_norm, dtype=tf.float32)
    wave = self.repr_to_t(mdct_norm_ft32)

    wave = tf.clip_by_value(wave, clip_value_min=-1., clip_value_max=1.)

    return wave
