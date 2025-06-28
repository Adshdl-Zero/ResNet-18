# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import layers # type: ignore

def load_cifar():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.

  data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
  ])

  return (x_train, y_train), (x_test, y_test), data_augmentation

