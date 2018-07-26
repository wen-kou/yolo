import unittest
import numpy as np

from yolov2.utils import bbox_selection


class MyTestCase(unittest.TestCase):
    def test_select_best_bounding_box(self):
        boxes = [[77.06828, 59.70646, 124.58709, 114.036606],
                 [67.42758, 440.51633, 163.79163, 684.22614],
                 [65.50801, 453.92377, 172.44803, 697.55664],
                 [81.69241, 440.677, 166.03093, 687.2009],
                 [81.72797, 462.19098, 167.25877, 693.87427],
                 [111.97971, 101.477325, 362.0512, 526.018],
                 [98.87349, 89.21441, 375.6418, 580.05237],
                 [128.16084, 90.084435, 398.1584, 541.00085],
                 [107.043915, 59.874847, 424.8609, 605.90857],
                 [119.165146, 227.19897, 411.6766, 606.626],
                 [114.581604, 81.05127, 466.3807, 553.8666],
                 [118.86455, 71.19839, 460.9149, 605.96606],
                 [135.35829, 181.85202, 445.3203, 580.5178],
                 [142.50552, 237.62746, 436.95386, 591.4086],
                 [120.57619, 61.082474, 507.3932, 565.6592],
                 [147.40495, 93.270996, 480.3167, 595.7992],
                 [150.13036, 174.36177, 472.65903, 589.60645],
                 [168.74147, 246.60379, 455.78787, 584.3927],
                 [198.82715, 144.4104, 514.62854, 308.95694],
                 [199.76602, 121.6929, 516.06714, 327.74042],
                 [197.13602, 139.61435, 551.52423, 314.09528],
                 [214.19843, 136.67719, 539.34515, 323.16492]]
        boxes = np.array(boxes, dtype=int)
        scores = [0.27970043, 0.21480599, 0.7869677, 0.34939542, 0.79514664,
                  0.1549603, 0.20520611, 0.6849882, 0.75120497, 0.13386242,
                  0.84201735, 0.8137847, 0.25504208, 0.3546458, 0.6363249,
                  0.6658478, 0.30635485, 0.24258769, 0.49034065, 0.5432607,
                  0.7126819, 0.7648384]
        classes = [3, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 16, 16, 16, 16]
        labels = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        res = bbox_selection.select_best_bounding_box(boxes, scores, classes)
        box = res[0][0]
        clzz = labels[res[2][0]]

        self.assertEqual('bicycle', clzz)
        self.assertEqual([114,  81, 466, 553], box.tolist())


if __name__ == '__main__':
    unittest.main()
