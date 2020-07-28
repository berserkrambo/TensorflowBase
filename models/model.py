# -*- coding: utf-8 -*-
# ---------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model


class DummyModel(Model):

    def __init__(self):
        super(DummyModel, self).__init__()

        # >> downsample blocks
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(2, 2), activation='relu')
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')
        self.conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')
        self.conv4 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')
        self.conv5 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')
        self.conv6 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(1, 1), activation='relu')
        # >> bottleneck
        self.flatten = keras.layers.Flatten(input_shape=(7, 7))
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dense2 = keras.layers.Dense(2, activation='softmax')


    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == '__main__':
    model = DummyModel()
    model.compile(loss='CategoricalCrossentropy')
    model.fit(x=tf.zeros((1,224,224,3)), y=tf.zeros((1,2)))
    model.summary()
