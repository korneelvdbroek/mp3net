# !/usr/bin/env python

"""Cross replica ops which allow basic model parallelism"""

import tensorflow as tf


@tf.function
def reduce_mean_over_replica_group(x, device_group_size):
  """
  Computes the mean over a group of replicas.
  Should be executed in the non-default replica context (i.e. on a specific replica of a tf.distributed.Strategy)

  :param x:                      A tensor
  :param device_group_size:      number of other replicas over which to take the average
  :return:                       tensor which is the average over device_group_size replicas which are in the same group

  :raises InvalidArgumentError:  when total number of devices in the replica context is not a multiple of device_group_size
  """
  replica_context = tf.distribute.get_replica_context()

  if replica_context is not None:
    x_ave = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, x) / tf.cast(device_group_size, dtype=x.dtype)
    return x_ave
  else:
    return x


def reduce_mean_over_replicas_tpu(x, devices_n):
  return tf.compat.v1.tpu.cross_replica_sum(x) / tf.cast(devices_n, dtype=x.dtype)
