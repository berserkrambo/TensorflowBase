from tensorflow import keras
from tf2_resnets import models


def get_model(input_shape, model_str, hm_ch):
    if model_str == "mobilenet-v2":
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
        )
    elif model_str == "resnet":
        base_model = models.ResNet18(input_shape=input_shape, weights='imagenet', include_top=False)

    elif model_str == "mobilenet-v1":
        base_model = keras.applications.MobileNet(
            input_shape=input_shape,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
        )

    # x = keras.layers.Conv2DTranspose(filters=64, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(
    #     base_model.output)
    # x = keras.layers.Conv2DTranspose(filters=64, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(x)
    # x = keras.layers.Conv2DTranspose(filters=64, kernel_size=4, padding='same', strides=(2, 2), activation='relu')(x)

    x = keras.layers.UpSampling2D()(base_model.output)
    x = keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', strides=(1,1), activation='relu')(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', strides=(1,1), activation='relu')(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', strides=(1,1), activation='relu')(x)

    # heatmap prediction
    hm = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    hm = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(hm)
    hm = keras.layers.Conv2D(filters=hm_ch, kernel_size=1, padding='same', strides=(1, 1), activation=None, name='hm')(hm)

    # size prediction
    s = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    s = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(s)
    s = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', strides=(1, 1), activation=None, name='s')(s)

    model = keras.Model(inputs=base_model.input, outputs=[hm, s])

    return model

if __name__ == '__main__':
    input_shape = (320,320,3)
    model = get_model(input_shape, "mobilenet-v1", 2)
    model.summary()
