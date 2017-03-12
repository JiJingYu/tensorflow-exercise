from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import tensorflow as tf

layers = tf.contrib.layers
learn = tf.contrib.learn


def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
  print(target.op.name, ' ', target.get_shape().as_list())

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, 28, 28, 1])
  print(feature.op.name, ' ', feature.get_shape().as_list())

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution2d(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    print(h_conv1.op.name, ' ', h_conv1.get_shape().as_list())
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1.op.name, ' ', h_pool1.get_shape().as_list())

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution2d(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    print(h_conv2.op.name, ' ', h_conv2.get_shape().as_list())
    h_pool2 = max_pool_2x2(h_conv2)
    print(h_pool2.op.name, ' ', h_pool2.get_shape().as_list())
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    print(h_pool2_flat.op.name, ' ', h_pool2_flat.get_shape().as_list())

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
  print(h_fc1.op.name, ' ', h_fc1.get_shape().as_list())

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  print(logits.op.name, ' ', logits.get_shape().as_list())
  loss = tf.losses.softmax_cross_entropy(target, logits)
  print(loss.op.name, ' ', loss.get_shape().as_list())

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op


def main(unused_args):
  ### Download and load MNIST dataset.
  mnist = learn.datasets.load_dataset('mnist')

  ### Linear classifier.
  feature_columns = learn.infer_real_valued_columns_from_input(
      mnist.train.images)
  classifier = learn.LinearClassifier(
      feature_columns=feature_columns, n_classes=10)
  classifier.fit(mnist.train.images,
                 mnist.train.labels.astype(np.int32),
                 batch_size=100,
                 steps=1000)
  y_true = mnist.test.labels
  y_pred = list(classifier.predict(mnist.test.images))

  classify_report = metrics.classification_report(y_true, y_pred)
  confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
  overall_accuracy = metrics.accuracy_score(y_true, y_pred)
  acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
  average_accuracy = np.mean(acc_for_each_class)
  score = metrics.accuracy_score(y_true, y_pred)
  print('classify_report : \n' , classify_report)
  print('confusion_matrix : \n', confusion_matrix)
  print('acc_for_each_class : \n', acc_for_each_class)
  print('average_accuracy: {0:f}'.format(average_accuracy))
  print('overall_accuracy: {0:f}'.format(overall_accuracy))
  print('score: {0:f}'.format(score))

  ### Convolutional network
  classifier = learn.Estimator(model_fn=conv_model)
  classifier.fit(mnist.train.images,
                 mnist.train.labels,
                 batch_size=100,
                 steps=20000)
  y_true = mnist.test.labels
  y_pred = list(classifier.predict(mnist.test.images))
  classify_report = metrics.classification_report(y_true, y_pred)
  confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
  overall_accuracy = metrics.accuracy_score(y_true, y_pred)
  acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
  average_accuracy = np.mean(acc_for_each_class)
  score = metrics.accuracy_score(y_true, y_pred)
  print('classify_report : \n', classify_report)
  print('confusion_matrix : \n', confusion_matrix)
  print('acc_for_each_class : \n', acc_for_each_class)
  print('average_accuracy: {0:f}'.format(average_accuracy))
  print('overall_accuracy: {0:f}'.format(overall_accuracy))
  print('score: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()
