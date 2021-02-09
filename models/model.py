# -*- coding: utf-8 -*-
# ---------------------

from tensorflow import keras


def get_MobileCenterModel(input_shape, max_pred=64):

    base_model = keras.applications.MobileNetV2(
    input_shape=input_shape,
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    )

    x = keras.layers.Conv2DTranspose(filters=input_shape[0]//2, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(base_model.output)
    x = keras.layers.Conv2DTranspose(filters=input_shape[0]//2, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2DTranspose(filters=input_shape[0]//2, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(x)

    # heatmap prediction
    hm = keras.layers.Conv2D(filters=input_shape[0]//4, kernel_size=3, padding='same', activation='relu')(x)
    hm = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', strides=(1,1), activation=None, name='hm')(hm)

    # center prediction
    hm_c = keras.layers.Conv2D(filters=max_pred, kernel_size=3, padding='same', activation='relu')(hm)
    hm_c = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', strides=(1, 1), activation=None, name='hm_c')(hm_c)

    # size prediction
    s = keras.layers.Conv2D(filters=max_pred, kernel_size=3, padding='same', activation='relu')(x)
    s = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', strides=(1, 1), activation=None, name='s')(s)

    # class prediction
    # c = keras.layers.Conv2D(filters=max_pred, kernel_size=3, padding='same', activation='relu')(x)
    # c = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', strides=(1, 1), activation=None, name='c')(c)

    model = keras.Model(inputs=base_model.input, outputs=[hm, hm_c, s])

    return model


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
    input_shape = (256,256,3)
    model = get_MobileCenterModel(input_shape)
    model.summary()

