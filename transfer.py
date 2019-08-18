import argparse
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging

LEARNING_RATE = 0.01


def shogi_inputs(data_dir, batch_size, image_size):
    training_count = 0
    for _ in tf.python_io.tf_record_iterator(os.path.join(data_dir, 'train.tfrecord')):
        training_count += 1

    def parser(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64)})
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)
        return image, features['label']

    t_dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'train.tfrecord'))
    t_dataset = t_dataset.map(parser)
    t_dataset = t_dataset.repeat()
    t_dataset = t_dataset.shuffle(batch_size * 10)
    t_dataset = t_dataset.batch(batch_size)

    v_dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'valid.tfrecord'))
    v_dataset = v_dataset.map(parser)
    v_dataset = v_dataset.repeat()
    v_dataset = v_dataset.batch(batch_size * 5)

    return [
        tf.compat.v1.data.make_initializable_iterator(t_dataset),
        tf.compat.v1.data.make_initializable_iterator(v_dataset),
        training_count
    ]


def run(args):
    with file_io.FileIO(os.path.join(args.data_dir, 'labels.txt'), 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    module = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/3')
    image_size = hub.get_expected_image_size(module)

    t_iter, v_iter, training_count = shogi_inputs(args.data_dir, args.batch_size, image_size)
    t_inputs, t_labels = t_iter.get_next()
    v_inputs, v_labels = v_iter.get_next()

    features = module(t_inputs)
    t_logits = tf.layers.dense(inputs=features, units=len(labels))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(t_labels, t_logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # Let's add an optimizer so we can train the network.
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss=cross_entropy_mean)

    with tf.compat.v1.Session() as sess:
        sess.run([
            tf.compat.v1.global_variables_initializer(),
            t_iter.initializer,
            v_iter.initializer,
        ])

        for i in range(100):
            print(sess.run([cross_entropy_mean, train_op]))


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
