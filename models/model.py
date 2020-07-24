# -*- coding: utf-8 -*-
# ---------------------

from tensorflow import keras
from tensorflow.keras import Model, Input

class DummyModel():

    def __init__(self):
        super(DummyModel, self).__init__()

        self.net = keras.Sequential(layers=[
            Input(shape=(224, 224, 3)),
            # >> downsample blocks
            keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(2,2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2,2), activation='relu'),
            # >> bottleneck
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(1,1), activation='relu'),
            # >> upsample blocks
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=(2,2), activation='relu'),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=(2,2), activation='relu'),
            # >> final conv
            keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', strides=(1,1), activation='relu')
            ]
        )


    def call(self, x):

        return self.net(x)


