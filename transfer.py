import argparse
import json
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


def dump_features(data_dir, filepath):
    model = mobilenet_v2()
    model.trainable = False
    model.build((None,) + IMAGE_SIZE + (3,))
    model.summary()
    batch_size = 32

    data = {
        'training': [],
        'validation': [],
        'testing': []
    }
    print('calculate features of training data')
    for images, labels in tfrecord_dataset(os.path.join(data_dir, 'training.tfrecord')).batch(batch_size):
        for features, label in zip(model(images), labels):
            data['training'].append([features.numpy().tolist(), int(label.numpy())])
        print({k: len(v) for k, v in data.items()})
    print('calculate features of validation data')
    for images, labels in tfrecord_dataset(os.path.join(data_dir, 'validation.tfrecord')).batch(batch_size):
        for features, label in zip(model(images), labels):
            data['validation'].append([features.numpy().tolist(), int(label.numpy())])
        print({k: len(v) for k, v in data.items()})
    print('calculate features of testing data')
    for images, labels in tfrecord_dataset(os.path.join(data_dir, 'testing.tfrecord')).batch(batch_size):
        for features, label in zip(model(images), labels):
            data['testing'].append([features.numpy().tolist(), int(label.numpy())])
        print({k: len(v) for k, v in data.items()})
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


class FeaturesSequence(tf.keras.utils.Sequence):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return np.array(self.data[idx][0], dtype=np.float32), np.array([self.data[idx][1]])

    def __len__(self):
        return len(self.data)


def run(args):
    if not os.path.exists(args.features):
        dump_features(args.data_dir, args.features)

    with open(args.features, 'r') as fp:
        data = json.load(fp)

    training_seq = FeaturesSequence(data['training'])
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(29, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None, 1280,))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'])
    model.fit_generator(training_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directories of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument(
        '--features',
        help='''Features file''',
        type=str,
        default='features.json')
    parser.add_argument(
        '--batch_size',
        help='''Batch size''',
        type=int,
        default=64)
    args = parser.parse_args()
    run(args)
