# -*- coding: utf-8 -*-
# ---------------------

from tensorflow import keras


def get_ResNet50(input_shape):

    resnet50 = keras.applications.resnet.ResNet50(input_shape=input_shape, include_top=False)

    # >> bottleneck
    x = keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), activation='relu')(resnet50.output)
    x = keras.layers.Conv2D(filters=128, kernel_size=2, strides=(1, 1), activation='relu')(x)
    outputs = keras.layers.Conv2D(filters=2, kernel_size=1, strides=(1, 1), activation='softmax')(x)

    outputs = keras.layers.Flatten()(outputs)

    return keras.Model(resnet50.inputs, outputs)


if __name__ == '__main__':
    input_shape = (128,128,3)
    model = get_ResNet50(input_shape)
    model.summary()

