import numpy as np
import pandas as pd
import os
import cv2
import copy
import argparse
import json
import tensorflow as tf

from yolov2.model import yolo_v2
from yolov2.data import data_proc
from yolov2.utils import parse_config
from yolov2.utils import bbox_selection


def main():

    resources_path = os.path.join(cur_dir_path, 'resources')
    cfg_path = os.path.join(resources_path, 'yolov2-voc.cfg')
    weights_path = os.path.join(resources_path, 'yolov2-voc_985591.weights')
    label_path = os.path.join(resources_path, 'labels.txt')
    cats_mapping_path = os.path.join(resources_path, 'cat_mapping_wt_l2.json')

    net_config, box_param = parse_config.parse_cfg(cfg_path)
    with open(cats_mapping_path, 'r') as fp:
        cats_mapping = json.load(fp)
    yolov2 = yolo_v2.YoloV2(net_config, weights_path, 20, speak_net=True)
    bbox_config = data_proc.BBoxRegression(box_param)
    labels = np.loadtxt(label_path)
    im_input_size = box_param['input_size']

    with tf.Graph().as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, im_input_size[0], im_input_size[1], 3])
        infer_op = yolov2.inference(input_tensor)
        bbox_op = bbox_config.get_bbox(infer_op)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for input_csv_file in input_csv_list:
                result = list()
                image_info = pd.read_csv(os.path.join(input_csv_folder_path, input_csv_file))
                columns = list(image_info.columns.values)
                im_path_key = columns[-1]
                items = image_info.iterrows()
                start = 0
                total_batch = int(np.ceil(len(image_info) / batch_size))
                flag = True
                remaining_count = len(image_info) - 1
                for i in range(total_batch):
                    batch_count = 0
                    image_list = list()
                    tmp_res = list()
                    tmp_columns = copy.copy(columns)
                    while True:
                        if remaining_count < 1:
                            break
                        remaining_count -= 1
                        if start < len(image_info):
                            if batch_count == batch_size - 1:
                                break
                            start += 1
                            item = next(items)[1]
                            im_path = item[im_path_key].replace('out/', '')
                            im = cv2.imread(os.path.join(input_folder_path, im_path))
                            if im is not None:
                                tmp = list()

                                for col in tmp_columns:
                                    tmp.append(item[col])
                                tmp_res.append(tmp)
                                image_list.append(im)
                                batch_count += 1
                            else:
                                flag = False
                    input_batch, im_size_list = data_proc.image_preproc(image_list, im_input_size)
                    bboxes = sess.run(bbox_op, feed_dict={input_tensor: input_batch})
                    bboxes = bboxes + (im_size_list, )
                    final_bboxes = bbox_selection.get_rect(*bboxes)
                    ress = bbox_selection.select_best_bounding_box_batch(final_bboxes)
                    for k, res in enumerate(ress):
                        clz = res[2]
                        clz = np.array(labels[clz], dtype=int)
                        tmp = list(ress[k])
                        tmp[2] = clz
                        ress[k] = tuple(tmp)
                    assert len(ress) == len(tmp_res)
                    tmp = list()
                    for k, res in enumerate(ress):
                        tmp_raw = tmp_res[k]
                        tmp_raw.extend(res)
                        tw_cat = cats_mapping[str(tmp_raw[0])]['id_l2']
                        tmp_raw.append(tw_cat)
                        tmp.append(tmp_raw)

                    result.extend(tmp)
                    print('Finish predict batch {} from file {}'.format(i, input_csv_file))
                    if flag is False:
                        tmp_columns.extend(['bbox', 'score', 'pred_cat', 'gt'])
                        df = pd.DataFrame(result, columns=tmp_columns)
                        df.to_csv(os.path.join(output_folder_path, input_csv_file))
                        break


if __name__ == '__main__':
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, default=os.path.join(cur_dir_path, 'resources/image_resources'), required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=16, required=False)
    parser.add_argument('-o', '--output_folder', type=str, default=os.path.join(cur_dir_path, 'result'), required=False)

    args = parser.parse_args()
    input_folder_path = args.input_folder
    batch_size = args.batch_size
    output_folder_path = args.output_folder
    if os.path.isdir(output_folder_path) is False:
        os.mkdir(output_folder_path)

    input_csv_folder_path = os.path.join(input_folder_path, 'convert')
    input_csv_list = os.listdir(input_csv_folder_path)
    input_csv_list = [input_csv_file for input_csv_file in input_csv_list if input_csv_file.endswith('.csv')]

    main()



