# !/usr/bin/env python

"""Common pathname manipulations to deal transparently with
1. local file system, as well as
2. Google Cloud Storage buckets (gs://)
"""

import os
import re
import glob
from google.cloud import storage
import tensorflow as tf

# install gcs filesystem on Windows:
# 1. bazel build //tensorflow/core/platform/cloud:gcs_file_system
#    use https://github.com/google/mediapipe/issues/724 to fix issues...
# 2. tf.load_library(path_to_the_above_dll)
#   see https://github.com/tensorflow/tensorflow/issues/38477
#tf.load_library(path_to_the_above_dll)


def is_gs(filepath):
  """
  :param filepath:  file path
  :return:          True, if filepath is for Google Cloud Storage, otherwise False
  """
  return filepath[0:5] == "gs://"


def join(path, *paths):
  if is_gs(path):
    # add '/' if needed
    sub_paths = [sub_path + ('' if sub_path[-1] == '/' else '/') for sub_path in [path] + list(paths)[:-1]]
    return ''.join(sub_paths) + paths[-1]
  else:
    return os.path.join(path, *paths)


def split(filepath):
  """Split full path in (folder, filename) """
  if is_gs(filepath):
    bucket, folder, filename = _split_gs_path(filepath)
    return "gs://" + bucket + "/" + folder + '/', filename
  else:
    if os.path.isdir(filepath):
      return filepath, None
    else:
      return os.path.split(filepath)


def findall(filepath_pattern):
  """Return list of all files which match the glob pattern"""
  if is_gs(filepath_pattern):
    # regex_pattern = _glob_to_re(filepath_pattern)
    # bucket = _get_bucket(filepath_pattern)
    # return [blob for blob in _list_gs_objects(bucket) if re.search(regex_pattern, blob) is not None]
    return tf.io.gfile.glob(filepath_pattern)
  else:
    return glob.glob(filepath_pattern)


def mkdir(path):
  """Makes new dir"""
  if not is_gs(path):
    return os.mkdir(path)
  return


def is_dir(filepath):
  """Checks if this is a directory that exists"""
  if is_gs(filepath):
    return tf.io.gfile.isdir(filepath)
  else:
    return os.path.isdir(filepath)


def write(filepath, data_str):
  """Write string to file"""
  if is_gs(filepath):
    bucket, folder, filename = _split_gs_path(filepath)
    _upload_blob(bucket, folder + "/" + filename, data_str)
  else:
    with open(filepath, 'w') as f:
      f.write(data_str)
  return


def _split_gs_path(filepath):
  """Breaks filepath into 1. gs:// 2. bucket 3. folder and 4. filename
  """
  assert is_gs(filepath), "{} is not a Google Cloud Storage path".format(filepath)

  filepath_no_prefix = filepath[5:]
  filepath_split = filepath_no_prefix.split('/')
  bucket = filepath_split[0]

  folder = '/'.join(filepath_split[1:-1]) if len(filepath_split) > 2 else ''
  filename = filepath_split[-1] if len(filepath_split) > 1 and filepath_split[-1] != '' else None

  return bucket, folder, filename


def _get_bucket(filepath):
  bucket, _, _ = _split_gs_path(filepath)
  return bucket


def _upload_blob(bucket_name, destination_blob_name, data_str):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)

  blob.upload_from_string(data_str)
