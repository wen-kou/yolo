import cv2
import numpy as np
import tensorflow as tf


def image_preproc(image_list, out_size):
    """
    :param image: ndarray BGR order
    :param out_size: tuple input size for net
    :return:
    """
    im_size_list = list()
    im_list = list()
    for image in image_list:
        if image is None:
            continue
        im_size_list.append((image.shape[0], image.shape[1]))
        im = cv2.resize(image, out_size)
        im = im / 255
        im = im[:, :, ::-1]
        im_list.append(im)

    return np.asarray(im_list), im_size_list


def get_rect(boxes, image_size_list):
    # height, width = float(original_image_size[0]), float(original_image_size[1])
    # im_dims = tf.stack([height, width, height, width], axis=0)
    # im_dims = tf.reshape(im_dims, [1, 4])
    # boxes = boxes * im_dims
    return


def _process_box(boxes, origin_image_size):
    """
    :param box: original bounding box 1D list of class, x_min, y_min, x_max, y_max.
    :param origin_image_size: tuple
    :return:
    """
    origin_image_size = np.array(origin_image_size)
    origin_image_size = np.expand_dims(origin_image_size, axis=0)
    boxes = np.reshape(boxes, (-1, 5))
    boxes_xy = [0.5 * (box[3:5] + box[1:3]) for box in boxes]
    boxes_wh = [box[3:5] - box[1:3] for box in boxes]
    boxes_xy = [boxxy / origin_image_size for boxxy in boxes_xy]
    boxes_wh = [boxwh / origin_image_size for boxwh in boxes_wh]
    boxes = [np.concatenate((boxes_xy[i][0], boxes_wh[i][0], box[0:1]), axis=0) for i, box in enumerate(boxes)]
    boxes = np.array(boxes)
    # find the max number of boxes
    max_boxes = 0
    for boxz in boxes:
        if boxz.shape[0] > max_boxes:
            max_boxes = boxz.shape[0]

    # add zero pad for training
    for i, boxz in enumerate(boxes):
        if boxz.shape[0] < max_boxes:
            zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
            boxes[i] = np.vstack((boxz, zero_padding))

    return np.array(boxes)


class BBoxRegression:
    def __init__(self, bbox_config):
        self.bbox_config = bbox_config
        self.n_classes = bbox_config['n_classes']
        self.input_W, self.input_H = bbox_config['input_size'][0], bbox_config['input_size'][1]
        assert self.input_W % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        assert self.input_H % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        self.out_W, self.out_H = bbox_config['out_size'][0], bbox_config['out_size'][1]

        self.anchors = bbox_config['anchors']
        self.n_classes = bbox_config['classes']
        self.n_anchors = len(bbox_config['anchors'])
        self.thresh = bbox_config['thresh']

    def get_regression_target(self, raw_bboxes, origin_image_size):
        """
        :param raw_bboxes: list of bounding boxes in one image
        :param origin_image_size: tuple
        :return:
        """
        true_boxes = _process_box(raw_bboxes, origin_image_size)
        detectors_mask, matching_true_boxes, label_tensor = self.preprocess_true_boxes(true_boxes)
        return detectors_mask, matching_true_boxes, label_tensor

    def preprocess_true_boxes(self, true_boxes):
        """Find detector in YOLO where ground truth box should appear."""

        num_anchors = len(self.anchors)
        conv_height = int(self.out_H)
        conv_width = int(self.out_W)

        detectors_mask = np.zeros(
            (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
        matching_true_boxes = np.zeros(
            (conv_height, conv_width, num_anchors, 4),
            dtype=np.float32)
        labels_true_boxes = np.zeros((conv_height, conv_width, num_anchors, self.n_classes), dtype=np.float32)

        for box in true_boxes:
            # scale box to convolutional feature spatial dimensions
            class_ind = int(box[4])
            box = box[0:4] * np.array(
                [conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box[1]).astype('int')
            j = np.floor(box[0]).astype('int')
            best_iou = 0
            best_anchor = 0
            for k, anchor in enumerate(np.array(self.anchors)):
                # Find IOU between box shifted to origin and anchor box.
                box_maxes = box[2:4] / 2.
                box_mins = -box_maxes
                anchor_maxes = (anchor / 2.)
                anchor_mins = -anchor_maxes

                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                detectors_mask[i, j, best_anchor] = 1
                adjusted_box = np.array(
                    [
                        box[0] - j, box[1] - i,
                        np.log(box[2] / self.anchors[best_anchor][0]),
                        np.log(box[3] / self.anchors[best_anchor][1])
                    ],
                    dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box
                labels_true_boxes[i, j, best_anchor, class_ind] = 1
        return matching_true_boxes, detectors_mask, labels_true_boxes

    def get_bbox(self, output):
        """
        :param output: tensor
        output from net [?, input_w/32, input_h/32, n_box*(objectness + coord + n_classes)]
        :param original_image_size tuple
        :return:
        """
        num_anchors = len(self.anchors)
        anchors_tensor = tf.reshape(tf.convert_to_tensor(self.anchors), [1, 1, 1, num_anchors, 2])

        conv_dims = tf.shape(output)[1:3]  # assuming channels last
        conv_height_index = tf.range(0, conv_dims[0])
        conv_width_index = tf.range(0, conv_dims[1])
        conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

        conv_width_index = tf.tile(
            tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = tf.reshape(tf.transpose(conv_width_index), [-1])
        conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index], axis=0))
        conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = tf.cast(conv_index, output.dtype.base_dtype.name)

        feats = tf.reshape(
            output, [-1, conv_dims[0], conv_dims[1], num_anchors, self.n_classes + 5])
        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), feats.dtype.base_dtype.name)

        box_xy = tf.nn.sigmoid(feats[..., 0:2])
        box_wh = tf.exp(feats[..., 2:4])
        box_confidence = tf.nn.sigmoid(feats[..., 4:5])
        box_class_probs = tf.nn.softmax(feats[..., 5:], axis=-1)

        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        boxes, scores, classes, pred_mask = self._post_proc(box_xy, box_wh, box_confidence, box_class_probs)
        return boxes, scores, classes, pred_mask

    def _post_proc(self, box_xy, box_wh, box_confidence, box_class_probs):
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        boxes = tf.concat([box_mins[..., 1:2], box_mins[..., 0:1],
                           box_maxes[..., 1:2], box_maxes[..., 0:1]], axis=-1)

        boxes_scores = box_confidence * box_class_probs
        box_classes = tf.argmax(boxes_scores, axis=-1)
        box_class_scores = tf.reduce_max(boxes_scores, axis=-1)
        pred_mask = box_class_scores >= self.thresh

        # pred_mask_tile = tf.expand_dims(pred_mask, 4)
        # pred_mask_tile = tf.tile(pred_mask_tile, [1, 1, 1, 1, 4])
        boxes = tf.boolean_mask(boxes, pred_mask)
        scores = tf.boolean_mask(box_class_scores, pred_mask)
        classes = tf.boolean_mask(box_classes, pred_mask)

        return boxes, scores, classes, pred_mask
