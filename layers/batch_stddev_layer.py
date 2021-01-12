# !/usr/bin/env python

"""Defines the batch stddev layer"""

import tensorflow as tf

import utils.tf_ops as tf_ops


class BatchStddevLayer(tf.keras.layers.Layer):
  """Layer to save the parameters of the filter (repr_convertor, drown and beta)
  """
  def __init__(self, batch_group_size, device_group_size, num_new_features=1):
    """

    :param batch_group_size:    Number of replicas to average over. Has to be a divisor of the
    :param device_group_size:   Number of replicas to average over.
                                Has to be either 1 (no average over devices), or the total number of replicas.
                                Values in-between do not give the right value!
    :param num_new_features:    Number of new feature to create with this layer
    """
    super(BatchStddevLayer, self).__init__()
    # should NOT be converted to tf.constant (tf.reshape expects fixed python constant)
    self.batch_group_size = batch_group_size
    self.device_group_size = device_group_size
    self.num_new_features = num_new_features
    self.s = None

  def build(self, input_shape):
    # [NHWC]  Input shape.
    self.s = input_shape
    pass

  @tf.function
  def call(self, inputs, **kwargs):
    y = tf.reshape(inputs, [-1, self.batch_group_size,
                            self.s[1], self.s[2],
                            self.s[3] // self.num_new_features, self.num_new_features])
    y = tf.cast(y, tf.float32)                               # D x [MG H W cn]  Cast to FP32

    y_ave = tf.reduce_mean(y, axis=1, keepdims=True)         # 1 x [M1 H W cn]  Mean over group G(=group_size) and devices D
    if self.device_group_size > 1:
      y_ave = tf_ops.reduce_mean_over_replica_group(y_ave, self.device_group_size)

    y = tf.square(y - y_ave)                                 # D x [MG H W cn]  Subtract the mean over group G(=group_size)

    y = tf.reduce_mean(y, axis=1)                            # . x [M  H W cn]  Calc variance over group G(=group_size) and devices D
    if self.device_group_size > 1:
      y = tf_ops.reduce_mean_over_replica_group(y, self.device_group_size)

    y = tf.sqrt(y + 1e-8)                                    # . x [M  H W cn]  Calc stddev over group G(=group_size)
    y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)     # . x [M  1 1 1n]  Take average of these stddev over fmaps (c) and pixels (HW) --> (M x n values)
    y = tf.squeeze(y, axis=[3])                              # . x [M  1 1  n]  Remove c-dimension

    y = tf.cast(y, inputs.dtype)                             # . x [M  1 1  n]  Cast back to original data type
    y = tf.tile(y, [self.batch_group_size,
                    self.s[1], self.s[2], 1])                # . x [N  H W  n]  Replicate over group and pixels (copy-paste same value to all features & pixels)
    return tf.concat([inputs, y], axis=3)                    # . x [N  H W  (C+n)]  Append as new fmap.

  def get_config(self):
    config = super(BatchStddevLayer, self).get_config()
    config.update({
      'batch_group_size': self.batch_group_size,
      'device_group_size': self.device_group_size,
      'num_new_features': self.num_new_features
    })
    return config
