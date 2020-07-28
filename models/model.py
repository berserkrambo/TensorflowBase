# -*- coding: utf-8 -*-
# ---------------------

from tensorflow import keras



def get_DummyModel(input_shape):

    inputs = keras.Input(input_shape)

    # >> downsample blocks
    x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(2, 2), activation='relu')(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=(1, 1), activation='relu')(x)

    # >> bottleneck
    x = keras.layers.Flatten(input_shape=(7, 7))(x)
    x = keras.layers.Dense(512, activation='relu')(x)

    outputs = keras.layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    input_shape = (224,224,3)
    model = get_DummyModel(input_shape)
    model.summary()

