import argparse
import os
import tensorflow as tf

from model import build_model
from dataset import tfrecord_dataset


def evaluate(data_dir, weights_path):
    with open(os.path.join(data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]

    model = build_model(len(labels))
    model.summary()
    model.load_weights(weights_path)
    model.trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    testing_data, testing_size = tfrecord_dataset(os.path.join(data_dir, 'testing.tfrecord'))
    test_result = model.evaluate(testing_data.batch(1), steps=testing_size)
    print(test_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--weights_path',
        help='''Path to file of trained weights''',
        type=str,
        default='')
    args = parser.parse_args()
    evaluate(args.data_dir, args.weights_path)
