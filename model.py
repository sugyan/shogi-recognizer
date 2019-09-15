import tensorflow as tf

INPUT_IMAGE_SIZE = (96, 96)


def mobilenet_v2():
    return tf.keras.applications.MobileNetV2(
        input_shape=INPUT_IMAGE_SIZE + (3,),
        include_top=False,
        pooling='avg',
        weights='imagenet')
