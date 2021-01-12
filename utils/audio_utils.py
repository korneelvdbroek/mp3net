# !/usr/bin/env python

"""Utility functions to work with audio files"""

import io
import numpy as np

import tensorflow as tf

# sound libraries
import librosa
import scipy.io.wavfile as wavfile
import soundfile


def load_audio(audio_file, sample_rate=None, mono=False, max_wav=0.999):
  """Loads and decodes audio file into 32-bit floating-point wav format

  Args:
    audio_file:            audio file
    sample_rate:           if specified, re-samples decoded audio to this rate
    max_wav:               normalizes audio signal, such that max_wav_value is the maximum wav value (default is 1.0)

  Returns:                 normalized (-1..1) audio sample  [samples_n, channels_n] and
                           output sample_rate
  """
  print('reading <{}>...'.format(audio_file))

  # allow download from gs://
  audio_file_gfile = tf.io.gfile.GFile(audio_file, mode='rb')
  audio_file_string = audio_file_gfile.read()
  audio_file_stream = io.BytesIO(audio_file_string)

  data, input_sample_rate = soundfile.read(audio_file_stream, dtype='float32')
  # librosa needs shape = (nb_channels, nb_samples)
  data = data.T
  wav = librosa.resample(data, input_sample_rate, sample_rate)

  # old code:
  # wav, sample_rate = librosa.core.load(audio_file_stream, sr=sample_rate, mono=mono)

  if wav.ndim == 2:
    wav = np.swapaxes(wav, 0, 1)

  assert wav.dtype == np.float32

  # At this point, wav is np.float32 either [nsamps,] or [nsamps, nch].
  # We want [samples_n, channels_n] to mimic 2D shape of spectral feats.
  if wav.ndim == 1:
    nsamps = wav.shape[0]
    actual_channels = 1
  else:
    nsamps, actual_channels = wav.shape
  wav = np.reshape(wav, [nsamps, actual_channels])

  # normalization needed: librosa does not normalize properly!!!
  # note: keep balance between stereo channels!
  factor = np.max(np.abs(wav))
  if factor > 0:
    wav = wav * max_wav / factor

  return wav, sample_rate


def save_audio(audio_filepath, wave_data, sample_rate, out_format='wav'):
  """Write audio data to file. The audio data should be in 32-bit floating-point wav format.

  :param audio_filepath:   filename to save the audio sample
  :param wave_data:        normalized (-1..1) sample data [samples_n, channels_n]
  :param sample_rate:      sample rate
  :param out_format        output audio file format
  :return:                 no return value
  """
  wave_data = np.clip(wave_data, -1.0, 1.0)
  if out_format.lower() == 'wav':
    wavfile.write(audio_filepath, sample_rate, wave_data)
  else:
    soundfile.write(audio_filepath, wave_data, sample_rate, out_format)
  return
