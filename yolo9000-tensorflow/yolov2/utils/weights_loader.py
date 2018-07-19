import numpy as np

from yolov2.utils import parse_config


def weights_loader(cfg_path, weights_path):
    """
    this script only load the .weights file trained by original implementation of yolo
    :param cfg_path:
    :param weights_path:
    :return:
    """
    layer_blks = parse_config.parse_cfg(cfg_path)
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(shape=(4,), dtype='int32', buffer=weights_file.read(16))

    # layer_blks.keys() immutable ?
    prev_layer_plane = []
    weights = dict()
    for i, layer_key in enumerate(layer_blks.keys()):
        if layer_key == 'input':
            prev_layer_plane.append(layer_blks[layer_key]['channels'])
            continue
        if 'convolutaional' in layer_key:
            tmp_dict = dict()
            filters = layer_blks[layer_key]['filters']
            size = layer_blks[layer_key]['size']

            batch_normalization = layer_blks[layer_key]['batch_normalization']
            weight_shape = (size, size, prev_layer_plane[-1], filters)

            prev_layer_plane.append(filters)
            weight_size = np.product(weight_shape)
            conv_bias = np.ndarray(shape=(filters,),
                                   dtype='float32',
                                   buffer=weights_file.read(filters * 4))
            tmp_dict.update({'convolutional_bias': conv_bias})

            if batch_normalization:
                bn_weights = np.ndarray(shape=(filters, 3),
                                        dtype='float32',
                                        buffer=weights_file.read(filters * 3))
                tmp_dict.update(
                    {'batch_normalization':
                         {'gamma': bn_weights[0],
                          'moving_mean': bn_weights[1],
                          'moving_var': bn_weights[2]}}
                )
            conv_weights = np.ndarray(
                shape=(filters, weight_shape[2], size, size),
                dtype='float32',
                buffer=weights_file.read(weight_size * 4)
            )
            conv_weights = conv_weights.transpose([2, 3, 1, 0])
            tmp_dict.update({'convolutional': conv_weights})

            weights.update({layer_key: tmp_dict})


