import numpy as np


def select_best_bounding_box(boxes, scores, classes, max_num=4):
    classes_unique = np.unique(classes)
    boxes_res = []
    scores_res = []
    classes_res = []
    for clz in classes_unique:
        indices = np.where(classes == clz)[0]
        if len(indices) == 1:
            boxes_res.extend(boxes[indices])
            scores_res.extend(np.array(scores)[indices])
            classes_res.append(clz)
            continue
        tmp_boxes = boxes[indices]
        tmp_scores = np.array(scores)[indices]
        tmp_scores_ranking = np.argsort(tmp_scores)
        tmp_boxes = tmp_boxes[tmp_scores_ranking]
        tmp_scores = tmp_scores[tmp_scores_ranking]
        picked = _nms(tmp_boxes)
        boxes_res.extend(tmp_boxes[picked])
        scores_res.extend(tmp_scores[picked])
        classes_res.extend([clz for i in range(len(picked))])

    return np.array(boxes_res, dtype=int), np.array(scores_res), np.array(classes_res)


def _nms(boxes, thresh=0.5):
    """
    :param boxes: ndarray
    :param thresh:
    :return:
    """
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    pick = []

    area = (y2 - y1 + 1) * (x2 - x1 + 1)
    indices = np.arange(0, boxes.shape[0])
    while indices.size != 0:
        last = indices.size
        i = last - 1
        pick.append(last - 1)
        suppression = [last - 1]
        for pos in range(last - 1):
            j = indices[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = xx2 - xx1 + 1
            h = yy2 - yy1 + 1

            if (w > 0) & (h > 0):
                overlap_area = w * h
                overlap = overlap_area / area[j]
                if overlap > thresh:
                    suppression.append(pos)
        indices = np.delete(indices, suppression)

    return pick


def get_rect(boxes, scores, classes, pred_mask, image_size_list):
    assert len(image_size_list) == len(pred_mask), 'size predict mask must be equal to that of image list'
    start = 0
    res = list()
    for i, image_size in enumerate(image_size_list):
        mask = pred_mask[i]
        res_len = len(np.where(mask == True)[0])
        image_size_tile = [image_size[0], image_size[1], image_size[0], image_size[1]]
        tmp_boxes = boxes[start: start + res_len] * image_size_tile
        tmp_boxes = np.array(tmp_boxes, dtype=int)
        tmp_scores = scores[start: start + res_len]
        tmp_clzz = classes[start: start + res_len]
        res.append([tmp_boxes, tmp_scores, tmp_clzz])

        start = start + res_len

    return res


