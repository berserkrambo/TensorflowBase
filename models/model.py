# -*- coding: utf-8 -*-
# ---------------------

from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential

class DummyModel(Sequential):

    def __init__(self, input_shape):
        super(DummyModel, self).__init__()

        self.add(Input(shape=input_shape))

        for i in self.get_layers():
            self.add(i)

        self.summary()

    def get_layers(self):
        net = [
            # >> downsample blocks
            keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu'),
            # >> bottleneck
            keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(1, 1), activation='relu'),
            keras.layers.Flatten(input_shape=(7,7)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(2, activation='softmax')


            # >> upsample blocks
            # keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=(2, 2),
            #                              activation='relu'),
            # keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=(2, 2),
            #                              activation='relu'),
            # # >> final conv
            # keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', strides=(1, 1), activation='relu')
        ]

        return net


if __name__ == '__main__':
    model = DummyModel(input_shape=(224,224,3))
