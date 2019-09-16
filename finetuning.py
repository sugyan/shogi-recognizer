import argparse
import os
import tensorflow as tf

from model import mobilenet_v2


def train(data_dir, weights_dir, batch_size=32):
    with open(os.path.join(data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]

    training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=2,
        width_shift_range=2,
        height_shift_range=2,
        brightness_range=(0.8, 1.2),
        channel_shift_range=0.2,
        zoom_range=0.02,
        rescale=1./255)
    training_data = training_datagen.flow_from_directory(
        os.path.join(data_dir, 'training'),
        target_size=(96, 96),
        classes=labels,
        batch_size=batch_size)

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    validation_data = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(96, 96),
        classes=labels,
        batch_size=batch_size)

    model = tf.keras.Sequential([
        mobilenet_v2(),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            len(labels),
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit_generator(
        training_data,
        epochs=100,
        validation_data=validation_data,
        callbacks=[
            tf.keras.callbacks.TensorBoard(),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(weights_dir, 'finetuning_weights-{epoch:02d}.h5'),
                save_weights_only=True),
        ])
    print(history.history)

    model.trainable = False
    model.save('finetuning_classifier.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--weights_dir',
        help='''Path to directory of weights files''',
        type=str,
        default='weights')
    parser.add_argument(
        '--batch_size',
        help='''Batch size''',
        type=int,
        default=32)
    args = parser.parse_args()
    train(args.data_dir, args.weights_dir, args.batch_size)
