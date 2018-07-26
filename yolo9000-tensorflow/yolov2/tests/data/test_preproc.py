import unittest
import cv2
import os
import tensorflow as tf
import numpy as np

from yolov2.data import data_proc
from yolov2.model import yolo_v2
from yolov2.utils import parse_config
from yolov2.utils import bbox_selection

labels = np.array(
    ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
     'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
     'teddy bear', 'hair drier', 'toothbrush'])


class MyTestCase(unittest.TestCase):
    # def test_something(self):
    #     config_path = '../../cfg/yolo.cfg'
    #     net_config, bbox_config = parse_config.parse_cfg(config_path)
    #     boxes = [[1, 10, 104, 150, 367],
    #              [2, 52, 25, 268, 510]]
    #     image_size = (450, 500)
    #     bbox_regression = data_proc.BBoxRegression(bbox_config)
    #     bbox_regression.get_regression_target(boxes, image_size)
    #
    #     self.assertEqual(True, False)

    def test_get_res(self):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.dirname(current_folder) + '/assets/cfg/yolo.cfg'
        net_config, bbox_param = parse_config.parse_cfg(config_path)
        weight_path = os.path.dirname(current_folder) + '/assets/bin/yolo.weights'
        image_path = os.path.dirname(current_folder) + '/assets/images'
        image_list = os.listdir(image_path)
        image_list = [image for image in image_list if image.endswith('.jpg')]
        images = [cv2.imread(os.path.join(image_path, image)) for image in image_list]
        test_input, im_size_list = data_proc.image_preproc(images, (608, 608))
        # test_input = np.expand_dims(image, axis=0)
        yolov2 = yolo_v2.YoloV2(net_config, weight_path, 16, speak_net=True)
        bbox_config = data_proc.BBoxRegression(bbox_param)
        with tf.Graph().as_default():
            input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 608, 608, 3])
            infer_op = yolov2.inference(input_tensor)
            bbox_op = bbox_config.get_bbox(infer_op)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                bbox = sess.run(bbox_op, feed_dict={input_tensor: test_input})

            bbox = bbox + (im_size_list,)
            final_bbox = bbox_selection.get_rect(*bbox)
            clzz1 = labels[np.unique(final_bbox[0][2])]
            clzz2 = labels[np.unique(final_bbox[1][2])]

        self.assertEqual(len(images), len(bbox[3]))
        self.assertEqual(['bicycle', 'motorbike', 'truck', 'dog'], clzz1.tolist())
        self.assertEquals(['person', 'dog', 'horse'], clzz2.tolist())


if __name__ == '__main__':
    unittest.main()
