#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:36:28 2019

@author: harini
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
#%%
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
#%%
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
#%%
def show_image(img):
    plt.imshow(np.squeeze(img), cmap='Greys_r')
testt=x_test[8]
show_image(testt)
#%%
soft=model.predict(x_test)
predictionf = tf.dtypes.cast(tf.argmax(soft, 1), tf.int32)
print(predictionf)
predictions=predictionf.numpy()
#%%
model.save_weights('./checkpoints/my_checkpoint')
model.load_weights('./checkpoints/my_checkpoint')


