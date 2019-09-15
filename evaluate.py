import argparse
import os
import tensorflow as tf


def evaluate(data_dir, model_path):
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as fp:
        labels = [line.strip() for line in fp.readlines()]

    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    testing_data = testing_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(96, 96),
        classes=labels)

    model = tf.keras.models.load_model(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.summary()

    test_result = model.evaluate(testing_data)
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
