# !/usr/bin/env python

"""Utility functions for Tensorflow"""

import numpy as np
import tensorflow as tf


def summary(model: tf.keras.Model, line_length=None, positions=None, print_fn=None, per_replica_batch=128):
  """Modified version of Model.summary() which only prints InputLayer and layers with weights"""
  if print_fn is None:
    print_fn = print

  # determine column positions
  line_length = line_length or 120
  positions = positions or [.45, .65, .85, 1.]
  if positions[-1] <= 1:
    positions = [int(line_length * p) for p in positions]
  # header names for the different log elements
  compute_type = tf.keras.mixed_precision.experimental.global_policy().compute_dtype

  to_display = ['Layer (type)', 'Output Shape', f'MiB ({compute_type})', 'Param #']
  formats = ['s', 's', '>11,.0f', '>12,']

  def print_row(fields, positions, formats):
    line = ''
    for i, (field, position, format) in enumerate(zip(fields, positions, formats)):
      if i > 0:
        line = line[:-1] + ' '
      line += f"{field:{format}}"
      line = line[:position]
      line += ' ' * (position - len(line))
    print_fn(line)

  print_fn('Model: "{}"'.format(model.name))
  print_fn('_' * line_length)
  print_row(to_display, positions, ['s']*4)
  print_fn('=' * line_length)

  def print_layer_summary(layer):
    """Prints a summary for a single layer.

    Arguments:
        layer: target layer.
    """
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = 'multiple'
    except RuntimeError:  # output_shape unknown in Eager mode.
      output_shape = '?'
    name = layer.name
    cls_name = layer.__class__.__name__

    output_shape_temp = output_shape[0] if type(output_shape) is list else output_shape
    if len(output_shape_temp) == 4:
      size = min(max(8, output_shape_temp[0]) * output_shape_temp[1] * output_shape_temp[2] * max(128, output_shape_temp[3]),
                 max(128, output_shape_temp[0]) * output_shape_temp[1] * output_shape_temp[2] * max(8, output_shape_temp[3]))
    else:
      size = (output_shape_temp[0] if output_shape_temp[0] is not None else 0) * np.prod([x for x in output_shape_temp[1:]], dtype=np.longlong)
    size *= tf.dtypes.as_dtype(compute_type).size
    size /= 2**20  # to MiB

    fields = [name + ' (' + cls_name + ')', str(output_shape), size, layer.count_params()]
    print_row(fields, positions, formats)

  for layer in model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer) or layer.get_weights():
      print_layer_summary(layer)
  print_fn('=' * line_length)

  def count_params(variables):
    return np.sum([np.prod(v.get_shape().as_list(), dtype=np.int32) for v in variables])

  trainable_count = count_params(model.trainable_weights)
  non_trainable_count = count_params(model.non_trainable_weights)

  print_fn('Total params: {:,}'.format(int(trainable_count + non_trainable_count)))
  print_fn('Trainable params: {:,}'.format(trainable_count))
  print_fn('Non-trainable params: {:,}'.format(non_trainable_count))
  print_fn('_' * line_length)
