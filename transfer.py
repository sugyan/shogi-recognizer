import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.platform import tf_logging as logging


def shogi_inputs(data_dir, batch_size, image_size):
    def parser(example):
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64)
        }
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)
        return image, features['label']

    return [
        tf.data.TFRecordDataset(os.path.join(data_dir, 'train.tfrecord')).map(parser),
        tf.data.TFRecordDataset(os.path.join(data_dir, 'valid.tfrecord')).map(parser),
    ]


def run(args):
    IMAGE_SHAPE = (224, 224)
    t_dataset, v_dataset = shogi_inputs(args.data_dir, args.batch_size, IMAGE_SHAPE)

    classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    model = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=(IMAGE_SHAPE)+(3,)),
    ])
    model.summary()

    data = []
    for images, labels in t_dataset.batch(args.batch_size):
        for features, label in zip(model(images), labels):
            data.append([features.numpy(), label.numpy()])
    np.save('train_features.npy', data)

    data = []
    for images, labels in v_dataset.batch(args.batch_size):
        for features, label in zip(model(images), labels):
            data.append([features.numpy(), label.numpy()])
    np.save('valid_features.npy', data)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directories of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument(
        '--batch_size',
        help='''Batch size''',
        type=int,
        default=64)
    args = parser.parse_args()
    run(args)
