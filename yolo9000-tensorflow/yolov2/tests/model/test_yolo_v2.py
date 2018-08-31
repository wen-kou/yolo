import unittest
import cv2
import os
import numpy as np
import tensorflow as tf

from yolov2.model import yolo_v2
from yolov2.utils import parse_config
from yolov2.data import data_proc


class MyTestCase(unittest.TestCase):
    def test_yolo_v2(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.dirname(current_dir) + '/assets/cfg/yolo.cfg'
        net_config, _ = parse_config.parse_cfg(config_path)
        yolov2 = yolo_v2.YoloV2(net_config, speak_net=True)
        with tf.Graph().as_default():
            with tf.Session():
                input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 608, 608, 3])
                infer_op = yolov2.inference(input_tensor)

        self.assertEqual([1, 19, 19, 425], infer_op.get_shape().as_list())

    def test_yolo_v2_with_weights(self):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.dirname(current_folder) + '/assets/cfg/yolo.cfg'
        net_config, bbox_param = parse_config.parse_cfg(config_path)
        weight_path = os.path.dirname(current_folder) + '/assets/bin/yolo.weights'
        image_path = os.path.dirname(current_folder) + '/assets/images/sample_dog.jpg'
        image = cv2.imread(image_path)
        test_input, im_size_list = data_proc.image_preproc([image], (608, 608))
        yolov2 = yolo_v2.YoloV2(net_config, weight_path, 16, speak_net=True)
        with tf.Graph().as_default():
            input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 608, 608, 3])
            infer_op = yolov2.inference(input_tensor)
            fe_op = yolov2.get_fe_opt()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                res = sess.run(infer_op, feed_dict={input_tensor: test_input})
                feature = sess.run(fe_op, feed_dict={input_tensor: test_input})

        self.assertEqual([1, 19, 19, 1024], list(feature.shape))
        self.assertEqual([1, 19, 19, 425], list(res.shape))


if __name__ == '__main__':
    unittest.main()
