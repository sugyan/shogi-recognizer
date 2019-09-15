import argparse
import os
import pathlib
import tensorflow as tf


def evaluate(data_dir, model_path):
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as fp:
        labels = [line.strip() for line in fp.readlines()]
    label_to_index = {label: index for index, label in enumerate(labels)}

    def load_image(image_path):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
        return tf.image.convert_image_dtype(image, tf.float32)

    image_paths = pathlib.Path(os.path.join(data_dir, 'test')).glob('*/*.jpg')
    image_paths = list(image_paths)
    label_index = [label_to_index[path.parent.name] for path in image_paths]
    images_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths]).map(load_image)
    labels_ds = tf.data.Dataset.from_tensor_slices(label_index)
    test_data = tf.data.Dataset.zip((images_ds, labels_ds)).shuffle(len(image_paths))

    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    test_result = model.evaluate(test_data.batch(1))
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
