import argparse
import glob
import os
import numpy as np
import tensorflow as tf

from model import mobilenet_v2


def dump_features(data_dir, features_dir):
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as fp:
        labels = [line.strip() for line in fp.readlines()]
    label_to_index = {label: index for index, label in enumerate(labels)}

    model = mobilenet_v2()
    model.trainable = False

    for category in sorted(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, category)):
            continue
        print(f'calculate features of {category} data...')

        features, label = [], []
        for root, dirs, files in os.walk(os.path.join(data_dir, category)):
            if not files:
                continue
            print(root)
            for filename in files:
                image = tf.io.read_file(os.path.join(root, filename))
                image = tf.io.decode_image(image, channels=3)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                features.append(model(tf.expand_dims(image, axis=0)).numpy().flatten())
                label.append(label_to_index[os.path.basename(root)])
        np.savez(os.path.join(features_dir, f'{category}.npz'), inputs=features, targets=label)


def train(data_dir, features_dir, weights_dir, batch_size=32):
    if len(glob.glob(os.path.join(features_dir, '*.npz'))) == 0:
        os.makedirs(features_dir, exist_ok=True)
        dump_features(data_dir, features_dir)

    def dataset(category):
        npz = np.load(os.path.join(features_dir, f'{category}.npz'))
        inputs = npz['inputs']
        targets = npz['targets']
        size = inputs.shape[0]
        return tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(size), size

    training_data, training_size = dataset('training')
    validation_data, validation_size = dataset('validation')

    with open(os.path.join(data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]
    classes = len(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((1280,)),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(
            classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(
        training_data.repeat().batch(batch_size),
        steps_per_epoch=training_size // batch_size,
        epochs=100,
        validation_data=validation_data.batch(batch_size),
        validation_steps=validation_size // batch_size,
        callbacks=[tf.keras.callbacks.TensorBoard()])
    print(history.history)

    model.save_weights(os.path.join(weights_dir, 'transfer_weights.h5'))
    classifier = tf.keras.Sequential([
        mobilenet_v2(),
        model,
    ])
    classifier.trainable = False
    classifier.save('transfer_classifier.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--features_dir',
        help='''Path to directory of features files''',
        type=str,
        default='features')
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
    train(args.data_dir, args.features_dir, args.weights_dir, args.batch_size)
