import argparse
import collections
import hashlib
import os
import random
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.mobilenet import mobilenet_v2
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging

# same as mobilenet_v1_train.py
_LEARNING_RATE_DECAY_FACTOR = 0.94


def shogi_inputs(args):
    training_count = 0
    for _ in tf.python_io.tf_record_iterator(os.path.join(args.data_dir, 'train.tfrecord')):
        training_count += 1

    def parser(example):
        features = tf.parse_single_example(example, {
            'image': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64)})
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [96, 96])
        return image, features['label']

    t_dataset = tf.data.TFRecordDataset(os.path.join(args.data_dir, 'train.tfrecord'))
    t_dataset = t_dataset.map(parser)
    t_dataset = t_dataset.repeat()
    t_dataset = t_dataset.shuffle(args.batch_size * 10)
    t_dataset = t_dataset.batch(args.batch_size)

    v_dataset = tf.data.TFRecordDataset(os.path.join(args.data_dir, 'valid.tfrecord'))
    v_dataset = v_dataset.map(parser)
    v_dataset = v_dataset.repeat()
    v_dataset = v_dataset.batch(args.batch_size * 5)

    return [
        t_dataset.make_initializable_iterator(),
        v_dataset.make_initializable_iterator(),
        training_count
    ]


def build_model(args):
    with file_io.FileIO(os.path.join(args.data_dir, 'labels.txt'), 'r') as f:
        labels = [label.strip() for label in f.readlines()]
    class_count = len(labels)

    g = tf.Graph()
    with g.as_default():
        t_iter, v_iter, training_count = shogi_inputs(args)
        t_inputs, t_labels = t_iter.get_next()
        v_inputs, v_labels = v_iter.get_next()
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
            t_logits, _ = mobilenet_v2.mobilenet(t_inputs, num_classes=class_count)
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=None)):
            v_logits, _ = mobilenet_v2.mobilenet(v_inputs, num_classes=class_count, reuse=True)
        # training
        tf.losses.sparse_softmax_cross_entropy(t_labels, t_logits)
        total_loss = tf.losses.get_total_loss(name='total_loss')
        num_epochs_per_decay = 2.5
        decay_steps = int(training_count / args.batch_size * num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
            0.045,
            tf.train.get_or_create_global_step(),
            decay_steps,
            _LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_tensor = slim.learning.create_train_op(
                total_loss,
                optimizer=tf.train.GradientDescentOptimizer(learning_rate))
        # validation accuracy
        indices = tf.argmax(v_logits, axis=1)
        correct = tf.equal(indices, v_labels)
        accuracy = tf.reduce_mean(tf.to_float(correct))

    slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
    slim.summaries.add_scalar_summary(accuracy, 'validation_accuracy', 'validation')
    return g, train_tensor, accuracy, t_iter.initializer, v_iter.initializer


def main(args=None):
    g, train_tensor, accuracy, t_init, v_init = build_model(args)

    # train step function
    def train_step(sess, train_op, global_step, train_step_kwargs):
        start_time = time.time()
        total_loss, np_global_step = sess.run([train_op, global_step])
        time_elapsed = time.time() - start_time
        # validation
        if np_global_step % 50 == 0:
            logging.info('validation accuracy: %.4f', sess.run(accuracy))

        if 'should_log' in train_step_kwargs:
            if sess.run(train_step_kwargs['should_log']):
                logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                             np_global_step, total_loss, time_elapsed)
        if 'should_stop' in train_step_kwargs:
            should_stop = sess.run(train_step_kwargs['should_stop'])
        else:
            should_stop = False
        return total_loss, should_stop

    # start training
    with g.as_default():
        init_op = tf.group(t_init, v_init, tf.global_variables_initializer())
        slim.learning.train(
            train_tensor,
            args.job_dir,
            graph=g,
            number_of_steps=args.number_of_steps,
            save_summaries_secs=args.save_summaries_secs,
            save_interval_secs=args.save_interval_secs,
            local_init_op=init_op,
            train_step_fn=train_step,
            global_step=tf.train.get_global_step())


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='''Directory for writing training checkpoints and logs''',
        type=str,
        default='logdir')
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
    parser.add_argument(
        '--number_of_steps',
        help='''Number of training steps to perform before stopping''',
        type=int)
    parser.add_argument(
        '--save_summaries_secs',
        help='''How often to save summaries, secs''',
        type=int,
        default=100)
    parser.add_argument(
        '--save_interval_secs',
        help='''How often to save checkpoints, secs''',
        type=int,
        default=100)
    main(parser.parse_args())
