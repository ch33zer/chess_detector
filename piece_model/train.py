import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflowjs as tfjs

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time

import pathlib

DATA_DIR="../piece_samples"
data_dir = pathlib.Path(DATA_DIR)

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
img_height = 64
img_width = 64
seed=123

#tf.debugging.set_log_device_placement(True)
train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Do this outside the model because tf.js doesn't support it
normalization_layer = layers.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

with tf.device('/device:cpu:0'):

  #data_augmentation = Sequential(
  #  [
  #  layers.RandomFlip("horizontal",
  #    input_shape=(img_height,
  #      img_width,
  #      3)),
  #  layers.RandomRotation(0.1),
  #  layers.RandomZoom(0.1),
  #  ]
  #  )

  model = Sequential([
    #data_augmentation,
    #layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

  model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  model.summary()

  epochs=25
  history = model.fit(
    normalized_train_ds,
    validation_data=normalized_val_ds,
    epochs=epochs
    )
  #history = model.fit(
  #  train_data,
  #  steps_per_epoch=2000,
  #  validation_data=validation_data,
  #  epochs=epochs,
  #  validation_steps=800)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#breakpoint()
# hacky, but allows the model to be loaded in javascript (we don't need the augmentation layers when we're just running the model anyways)
#del model.layers[0]

model.save('model')
tfjs.converters.save_keras_model(model, 'modeljs')


epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()