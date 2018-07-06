import collections
import hashlib
import os
import random
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.mobilenet import mobilenet_v2
from tensorflow.python.platform import tf_logging as logging

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_dir', os.path.join(os.path.dirname(__file__), '..', 'dataset'),
                           '''Path to directories of labeled images''')
tf.app.flags.DEFINE_string('checkpoint_dir', 'logdir',
                           '''Directory for writing training checkpoints and logs''')
tf.app.flags.DEFINE_integer('validation_percentage', 10,
                            'What percentage of images to use as a validation set')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            '''Batch size''')
tf.app.flags.DEFINE_integer('number_of_steps', None,
                            '''Number of training steps to perform before stopping''')
tf.app.flags.DEFINE_integer('save_summaries_secs', 100,
                            '''How often to save summaries, secs''')
tf.app.flags.DEFINE_integer('save_interval_secs', 100,
                            '''How often to save checkpoints, secs''')

# same as mobilenet_v1_train.py
_LEARNING_RATE_DECAY_FACTOR = 0.94


def create_image_lists(image_dir, validation_percentage):
    result = collections.OrderedDict()
    sub_dirs = [d for d in tf.gfile.ListDirectory(image_dir) if tf.gfile.IsDirectory(os.path.join(image_dir, d))]
    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(image_dir, dir_name, '*.jpg')
        file_list.extend(tf.gfile.Glob(file_glob))
        training_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
            hashed = hashlib.sha1(tf.compat.as_bytes(file_name)).hexdigest()
            percentage_hash = int(hashed, 16) % 100
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)
        result[dir_name] = {
            'training': training_images,
            'validation': validation_images,
        }
    return result


def shogi_inputs(image_lists):
    class_count = len(image_lists.keys())
    t_count, v_count = 0, 0
    for l in image_lists.values():
        t_count += len(l['training'])
        v_count += len(l['validation'])

    def generate_dataset(category):
        images = []
        labels = []
        label_names = []
        for label_index in range(class_count):
            label_name = list(image_lists.keys())[label_index]
            label_names.append(label_name)
            category_list = image_lists[label_name][category]
            for basename in category_list:
                images.append(os.path.join(FLAGS.image_dir, label_name, basename))
                labels.append(label_index)
        zipped = list(zip(images, labels))
        random.shuffle(zipped)
        return tf.data.Dataset.from_tensor_slices((
            [e[0] for e in zipped],
            [e[1] for e in zipped]))

    def parser(file_path, label_index):
        image = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [96, 96])
        return image, tf.to_int64(label_index)

    t_dataset = generate_dataset('training')
    t_dataset = t_dataset.map(parser)
    t_dataset = t_dataset.repeat()
    t_dataset = t_dataset.shuffle(FLAGS.batch_size * 10)
    t_dataset = t_dataset.batch(FLAGS.batch_size)

    v_dataset = generate_dataset('validation')
    v_dataset = v_dataset.map(parser)
    v_dataset = v_dataset.repeat()
    v_dataset = v_dataset.batch(FLAGS.batch_size * 5)

    return [
        t_dataset.make_initializable_iterator(),
        v_dataset.make_initializable_iterator(),
        t_count,
    ]


def build_model():
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())

    g = tf.Graph()
    with g.as_default():
        t_iter, v_iter, training_count = shogi_inputs(image_lists)
        t_inputs, t_labels = t_iter.get_next()
        v_inputs, v_labels = v_iter.get_next()
        with slim.arg_scope(mobilenet_v2.training_scope()):
            t_logits, _ = mobilenet_v2.mobilenet(t_inputs, num_classes=class_count)
            v_logits, _ = mobilenet_v2.mobilenet(v_inputs, num_classes=class_count, reuse=True)
        # training
        tf.losses.sparse_softmax_cross_entropy(t_labels, t_logits)
        total_loss = tf.losses.get_total_loss(name='total_loss')
        num_epochs_per_decay = 2.5
        decay_steps = int(training_count / FLAGS.batch_size * num_epochs_per_decay)
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
    return g, train_tensor, accuracy, t_iter.initializer, v_iter.initializer, list(image_lists.keys())


def main(args=None):
    g, train_tensor, accuracy, t_init, v_init, labels = build_model()
    # save labels
    with open(os.path.join(FLAGS.checkpoint_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            print(label, file=f)

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
            FLAGS.checkpoint_dir,
            graph=g,
            number_of_steps=FLAGS.number_of_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs,
            local_init_op=init_op,
            train_step_fn=train_step,
            global_step=tf.train.get_global_step())


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf.app.run(main)
