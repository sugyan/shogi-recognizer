import tensorflow as tf

INPUT_IMAGE_SIZE = (96, 96)


def build_model(classes):
    return tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=INPUT_IMAGE_SIZE + (3,),
            include_top=False,
            pooling='avg',
            weights='imagenet'),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    ])
