import numpy as np
import tensorflow as tf
from PIL import Image
from nets.mobilenet import mobilenet_v2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', 'logdir/model.checkpoint',
                           '''Path to checkpoint file''')
tf.app.flags.DEFINE_string('labels', 'labels.txt',
                           '''Path to output labels file''')
tf.app.flags.DEFINE_string('input_image', '',
                           '''Path to input image file''')
IMAGE_SIZE = 96


class Recognizer:
    def __init__(self, checkpoint_path, labels):
        with tf.gfile.Open(labels) as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.g = tf.Graph()
        with self.g.as_default():
            self.placeholder = tf.placeholder(tf.float32, shape=(None, 96, 96, 3))
            logits, _ = mobilenet_v2.mobilenet(self.placeholder, len(self.labels))
            self.top3 = tf.nn.top_k(logits, 3)
        self.checkpoint_path = checkpoint_path

    def run(self, input_image):
        img = Image.open(input_image).convert('RGB')
        h, w = img.height / 9, img.width / 9
        inputs = []
        for file in range(1, 10):
            for rank in range(1, 10):
                cropped = img.crop([
                    w * (9 - file),
                    h * (rank - 1),
                    w * (10 - file),
                    h * rank])
                resized = cropped.resize([IMAGE_SIZE, IMAGE_SIZE], resample=Image.BILINEAR)
                inputs.append(np.array(resized) / 255.0)
        with self.g.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, self.checkpoint_path)
                results, indices = sess.run(self.top3, feed_dict={self.placeholder: inputs})
        # for result in results:
        #     print(result)
        # board = [[]]
        for rank in range(9):
            row = ''
            for file in range(9):
                i = (8 - file) * 9 + rank
                best = indices[i][0]
                label = self.labels[best]
                s = label.split('_')
                if len(s) == 1:
                    if label == 'BLANK':
                        row += ' * '
                else:
                    if s[0] == 'B':
                        row += '+'
                    if s[0] == 'W':
                        row += '-'
                    row += s[1]
            print(row)


if __name__ == '__main__':
    Recognizer(FLAGS.checkpoint_path, FLAGS.labels).run(FLAGS.input_image)
