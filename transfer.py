import argparse
import glob
import os
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (96, 96)


def mobilenet_v2():
    return tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        pooling='avg',
        weights='imagenet')


def tfrecord_dataset(filepath):
    def parser(example):
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64)
        }
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, features['label']

    return tf.data.TFRecordDataset(filepath).map(parser)


def dump_features(data_dir, features_dir):
    model = mobilenet_v2()
    model.trainable = False
    model.build((None,) + IMAGE_SIZE + (3,))
    model.summary()
    batch_size = 50

    for filepath in glob.glob(os.path.join(data_dir, '*.tfrecord')):
        category = os.path.splitext(os.path.basename(filepath))[0]
        print(f'calculate features of {category} data', end='', flush=True)
        inputs, targets = [], []
        for images, labels in tfrecord_dataset(filepath).batch(batch_size):
            for features, label in zip(model(images), labels):
                inputs.append(features.numpy())
                targets.append(label.numpy())
            print('.', end='', flush=True)
        print()
        np.savez(os.path.join(features_dir, f'{category}.npz'), inputs=inputs, targets=targets)


def run(args):
    with open(os.path.join(args.data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]

    if len(glob.glob(os.path.join(args.features_dir, '*.npz'))) == 0:
        os.makedirs(args.features_dir, exist_ok=True)
        dump_features(args.data_dir, args.features_dir)

    def dataset(category):
        npz = np.load(os.path.join(args.features_dir, f'{category}.npz'))
        inputs = npz['inputs']
        targets = npz['targets']
        size = inputs.shape[0]
        return tf.data.Dataset.from_tensor_slices((inputs, targets)) \
            .shuffle(inputs.shape[0]) \
            .batch(args.batch_size), size

    training_data, training_size = dataset('training')
    validation_data, _ = dataset('validation')
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            len(labels), activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    ])
    model.build((None, 1280))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    history = model.fit(
        training_data.repeat(),
        epochs=50,
        steps_per_epoch=training_size // args.batch_size,
        validation_data=validation_data,
        verbose=2)

    testing_data, _ = dataset('testing')
    test_result = model.evaluate(testing_data, verbose=0)
    print(history.history, test_result)

    model.save_weights(os.path.join(args.weights_dir, 'transfer.h5'))


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
    run(args)
