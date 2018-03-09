# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# Process images of this size. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 64

# Global constants describing the chess-cnn dataset
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 180000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 30000

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def read_labeled_image_list(image_list_file):
  """Reads a .txt file containing pathes and labeles
  Args:
    image_list_file: a .txt file with one /path/to/image per line
    label: optionally, if set label will be pasted after each line
  Returns:
    List with all filenames in file image_list_file
  """
  f = open(image_list_file, 'r')
  filenames = []
  labels = []
  for line in f:
    filename, label = line[:-1].split(' ')
    filenames.append(filename)
    labels.append(int(label))
  f.close()
  return filenames, labels

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string.
  Args:
    filename_and_label_tensor: A scalar string tensor.
  Returns:
    Two tensors: the decoded image, and the string label.
  """
  label = input_queue[1]
  file_contents = tf.read_file(input_queue[0])
  example = tf.image.rgb_to_grayscale(tf.image.decode_png(file_contents))
  return example, label

def inputs(which_data, data_dir, batch_size):
  """Construct input for chess-cnn using the Reader ops.

  Args:
    which_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the chess-cnn data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # TODO: add directory
  
  if not which_data:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    filename = '/home/arash/Software/repositories/chesscnn/data/images/s64/train.txt'
    print('training')
  else:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    filename = '/home/arash/Software/repositories/chesscnn/data/images/s64/test.txt'
    print('testing')

  with tf.name_scope('input'):
  
    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list(filename)
    
    images = tf.convert_to_tensor(image_list, tf.string)
    labels = tf.convert_to_tensor(label_list, tf.int32)
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels])
    
    image, label = read_images_from_disk(input_queue)
    
    float_image = tf.cast(image, tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(float_image)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    tf.reshape(label, [])
#    label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
