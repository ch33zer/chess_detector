import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflowjs as tfjs
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time

import pathlib

DATA_DIR="../piece_samples"
data_dir = pathlib.Path(DATA_DIR)

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
img_height = 96
img_width = 96
img_size=(img_height, img_width)
img_shape=img_size + (3,)
seed=123
DEBUG=False
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)] if DEBUG else []

#tf.debugging.set_log_device_placement(True)
train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=img_size,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=img_size,
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
normalization_layer = layers.Rescaling(1./127.5, offset=-1)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

with tf.device('/device:gpu:0'):

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
  base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape,
    include_top=False,
    weights='imagenet')
  base_model.trainable=False
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = layers.Dense(num_classes)
  inputs = tf.keras.Input(shape=img_shape)
  x = base_model(inputs, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  model.summary()

  initial_epochs=30
  history = model.fit(normalized_train_ds,
    epochs=initial_epochs,
    validation_data=normalized_val_ds,
    callbacks = tensorboard_callback)  #history = model.fit(
  #  train_data,
  #  steps_per_epoch=2000,
  #  validation_data=validation_data,
  #  epochs=epochs,
  #  validation_steps=800)

  print("Fine tuning...")
  base_model.trainable = True

  # Fine-tune from this layer onwards
  fine_tune_at = 100

  # Freeze all the layers before the `fine_tune_at` layer
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  fine_tune_epochs = 10
  total_epochs =  initial_epochs + fine_tune_epochs

  history_fine = model.fit(normalized_train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=normalized_val_ds,
                         callbacks = tensorboard_callback)



acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

#breakpoint()
# hacky, but allows the model to be loaded in javascript (we don't need the augmentation layers when we're just running the model anyways)
#del model.layers[0]

model.save('model')
tfjs.converters.save_keras_model(model, 'modeljs')


epochs_range = range(initial_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
