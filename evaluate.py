import argparse
import os
import tensorflow as tf

from dataset import tfrecord_dataset


def evaluate(data_dir, model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    testing_data, testing_size = tfrecord_dataset(os.path.join(data_dir, 'testing.tfrecord'))
    test_result = model.evaluate(testing_data.shuffle(testing_size).batch(1), steps=testing_size)
    print(test_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--model_path',
        help='''Path to file of trained models''',
        type=str,
        default='')
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_path)
