import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from nets.mobilenet import mobilenet_v2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graph', 'output_graph.pb',
                           '''Path to graph file''')
tf.app.flags.DEFINE_string('labels', 'labels.txt',
                           '''Path to labels file''')
tf.app.flags.DEFINE_string('input_image', '',
                           '''Path to input image file''')
IMAGE_SIZE = 96


class Recognizer:
    def __init__(self, graph, labels):
        with tf.gfile.Open(labels) as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.graph_def = tf.GraphDef()
        with tf.gfile.Open(graph, 'rb') as f:
            self.graph_def.ParseFromString(f.read())

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
                resized = cropped.resize([IMAGE_SIZE, IMAGE_SIZE])
                inputs.append(np.array(resized) / 255.0)
        with tf.Graph().as_default() as g:
            tf.import_graph_def(self.graph_def, name='')
            placeholder = g.get_tensor_by_name('MobilenetV2/input:0')
            final_result = g.get_tensor_by_name('MobilenetV2/Logits/output:0')
            top3 = tf.nn.top_k(final_result, 3)
            with tf.Session() as sess:
                results, indices = sess.run(top3, feed_dict={placeholder: inputs})
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
    Recognizer(FLAGS.graph, FLAGS.labels).run(FLAGS.input_image)
