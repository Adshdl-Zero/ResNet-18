# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras

def residual_block(x, filter, stride = 1, use_projection = False):
  shortcut = x
  x = keras.layers.Conv2D(filter, (3,3), strides = stride, padding = 'same', use_bias = False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.Conv2D(filter, (3,3), strides = 1, padding = 'same', use_bias = False)(x)
  x = keras.layers.BatchNormalization()(x)
  if use_projection:
    shortcut = keras.layers.Conv2D(filter, (1,1), strides = stride, padding = 'same', use_bias = False)(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)
  x = keras.layers.Add()([x, shortcut])
  x = keras.layers.ReLU()(x)
  return x

def res_net(input_shape = (32, 32, 3), num_classes = 10):
  inputs = keras.layers.Input(shape = input_shape)
  x = keras.layers.Conv2D(64, (3, 3), strides = 1, padding = 'same', use_bias = False)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)

  x = residual_block(x, 64)
  x = residual_block(x, 64)

  x = residual_block(x, 128, 2, True)
  x = residual_block(x, 128)

  x = residual_block(x, 256, 2, True)
  x = residual_block(x, 256)

  x = residual_block(x, 512, 2, True)
  x = residual_block(x, 512)

  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(num_classes, activation = 'softmax')(x)

  model = keras.models.Model(inputs, outputs)
  return model
