# !/usr/bin/env python

"""Utility functions to load data into model"""

import tensorflow as tf


@tf.function
def slice_into_fixed_length(audio, slice_len, slice_hop, slice_first_only=False):
  """
  Slice up audio into fixed-length fragments

  :param audio:                (un-batched) tensor encoding audio data [blocks_n, freqs_n, channels_n, 2]
  :param slice_len:            block length of output slices
  :param slice_hop:            number of blocks in-between slices
                               Note: don't set this too low, as it will explode the dataset
  :param slice_first_only:     if True, then only first slice is returned. Defaults to False
  :return:                     sequence of tensors of shape [slice_len, freqs_n, channels_n, 2]
  """
  # note: batches_n dimension is not present!
  if slice_hop < 1:
    raise ValueError('Overlap ratio too high')

  blocks_n = tf.shape(audio)[0]

  # Extract sliceuences [blocks_n, freqs_n, channels_n, data/noise]
  audio_slices = tf.map_fn(fn=lambda start: audio[start:start+slice_len, :, :, :],
                           elems=tf.range(0, blocks_n-slice_len+1, slice_hop),
                           fn_output_signature=audio.dtype)

  # Only use first slice if requested
  if slice_first_only:
    audio_slices = audio_slices[0:1, :, :, :, :]

  return audio_slices


def load_dataset(
    distributed_context,
    input_filenames,
    global_batch_size,
    slice_len,
    slice_hop,
    freq_n, channels_n,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    tpu=False):
  """Decodes audio file paths into mini-batches of samples.

  Args:
    distributed_context: input_context class containing information on how we can pipe data into the distributed cluster
    input_filenames: filename and path of preprocessed audio file path
    global_batch_size: Number of items in the batch.
    repeat: If true (for training), continuously iterate through the dataset.
    shuffle: If true (for training), buffer and shuffle the sliceuences.
    shuffle_buffer_size: Number of examples to queue up before grabbing a batch.
    tpu: Flag to indicate whether we run on a TPU (default = False)

  Returns:
    A dataset with tensor of the form [batches_n, blocks_n, freqs_n, channels_n=1]
  """
  # Create dataset of filepaths (still on CPU)
  # dataset = tf.data.TFRecordDataset(input_filenames, compression_type="GZIP",
  #                                   num_parallel_reads=tf.data.experimental.AUTOTUNE)
  # note: if file is TOO small (e.g. needs to load 2x to fill one batch) then CPU might become the bottleneck (gzip...)
  dataset = tf.data.Dataset.from_tensor_slices(input_filenames)

  # - Be sure to shard before you use any randomizing operator (such as shuffle).
  # - Generally it is best if the shard operator is used early in the dataset
  #   pipeline. For example, when reading from a set of TFRecord files, shard
  #   before converting the dataset to input samples. This avoids reading every
  #   file on every worker. The following is an example of an efficient
  #   sharding strategy within a complete pipeline:
  dataset = dataset.shard(distributed_context.num_input_pipelines, distributed_context.input_pipeline_id)

  # open files (they contain string-ized tensors)
  dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE,
                               deterministic=False)

  def read_and_cast(x):
    # convert string-ized tensors to float32 tensors
    x = tf.io.parse_tensor(x, tf.float32)
    # convert to keras compute_dtype
    x = tf.cast(x, dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)
    return x
  dataset = dataset.map(read_and_cast, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def filter_short_samples(audio):
    # only keep samples which are long enough
    return tf.shape(audio)[0] >= slice_len
  dataset = dataset.filter(filter_short_samples)

  # slice on the fly (otherwise too much data to save to GCS and e-gress...)
  def _slice_file_wrapper(audio):
    # set shape [blocks_n, freqs_n, channels_n=1, data/masking]
    audio.set_shape([None, freq_n, channels_n, 2])

    audio_slices = slice_into_fixed_length(audio, slice_len=slice_len, slice_hop=slice_hop)
    return tf.data.Dataset.from_tensor_slices(audio_slices)
  dataset = dataset.interleave(_slice_file_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # I prefer repeat & shuffle to be done on the individual tensors,
  # not on the inputfiles (since then order in file is always the same)
  if repeat: dataset = dataset.repeat()

  if shuffle: dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  # Make batches (each worker will chew off a per_worker_batch_size)
  per_worker_batch_size = distributed_context.get_per_replica_batch_size(global_batch_size)
  print("Loading dataset:")
  print("  batch_size = {}".format(global_batch_size))
  print("  per_worker_batch_size = {}".format(per_worker_batch_size))

  dataset = dataset.batch(per_worker_batch_size, drop_remainder=True)

  # this starts fetching on host (cpu) while device (gpu/tpu) is still training previous batch
  dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset