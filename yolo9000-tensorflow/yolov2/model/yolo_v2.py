import tensorflow as tf
import numpy as np
import json

from yolov2.utils import parse_config


def _get_weights_from_bytes(weights_file, layer_block, prev_layer_plane):
    byte_count = 0
    conv_block_weights_dict = dict()
    filters = layer_block['filters']
    size = layer_block['size']

    batch_normalization = layer_block['batch_normalize']
    weight_shape = (size, size, prev_layer_plane, filters)

    weight_size = np.product(weight_shape)
    conv_bias = np.ndarray(shape=(filters,),
                           dtype='float32',
                           buffer=weights_file.read(filters * 4))
    byte_count += filters * 4

    if batch_normalization:
        bn_weights = np.ndarray(shape=(3, filters),
                                dtype='float32',
                                buffer=weights_file.read(filters * 12))
        byte_count += filters * 12
        conv_block_weights_dict.update(
            {'batch_normalization':
                 {'gamma': bn_weights[0],
                  # 'beta': conv_bias,
                  'moving_mean': bn_weights[1],
                  'moving_var': bn_weights[2]}}
        )
        conv_block_weights_dict.update({'convolutional_bias': conv_bias})
    else:
        conv_block_weights_dict.update({'convolutional_bias': conv_bias})

    conv_weights = np.ndarray(
        shape=(filters, weight_shape[2], size, size),
        dtype='float32',
        buffer=weights_file.read(weight_size * 4)
    )
    byte_count += weight_size * 4
    conv_weights = conv_weights.transpose([2, 3, 1, 0])
    conv_block_weights_dict.update({'convolutional': conv_weights})

    return conv_block_weights_dict, byte_count


def _variable_initial(name, shape, initializer=None):
    if initializer is None:
        stddev = 5e-2
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    weights = tf.get_variable(name=name,
                              shape=shape,
                              initializer=initializer,
                              dtype=tf.float32,
                              trainable=True)
    return weights


def _build_conv(inp_tensor, pad_val, filter_size, filters, stride, filter_ini=None):
    out_tensor = tf.nn.conv2d(inp_tensor,
                              filter=_variable_initial('filter',
                                                       shape=[filter_size, filter_size, inp_tensor.get_shape()[3],
                                                              filters],
                                                       initializer=filter_ini),
                              strides=[1, stride, stride, 1],
                              padding=pad_val)
    # bias = _variable_initial(name='bias', shape=[filters])
    return out_tensor


def _build_bias(filters, filter_ini=None):
    bias = _variable_initial(name='bias', shape=[filters], initializer=filter_ini)
    return bias


def _conv(inp_tensor, scope, conv_layer_block, conv_block_weights_ini=None):
    with tf.variable_scope(scope):
        pad_val = conv_layer_block['pad']
        filter_size = int(conv_layer_block['size'])
        filters = int(conv_layer_block['filters'])
        stride = int(conv_layer_block['stride'])
        activate = conv_layer_block['activation']
        if conv_layer_block['batch_normalize']:
            if conv_block_weights_ini is None:
                out_tensor = _build_conv(inp_tensor,
                                         pad_val,
                                         filter_size,
                                         filters,
                                         stride)
                out_tensor = tf.layers.batch_normalization(out_tensor)
                bias = _build_bias(filters)
                out_tensor = tf.nn.bias_add(out_tensor, bias)
            else:
                out_tensor = _build_conv(inp_tensor,
                                         pad_val,
                                         filter_size,
                                         filters,
                                         stride,
                                         filter_ini=tf.constant_initializer(conv_block_weights_ini['convolutional']))
                out_tensor = tf.layers.batch_normalization(out_tensor,
                                                           gamma_initializer=tf.constant_initializer(
                                                               conv_block_weights_ini['batch_normalization']['gamma']),
                                                           # beta_initializer=tf.constant_initializer(conv_block_weights_ini['batch_normalization']['beta']),
                                                           moving_mean_initializer=tf.constant_initializer(
                                                               conv_block_weights_ini['batch_normalization'][
                                                                   'moving_mean']),
                                                           moving_variance_initializer=tf.constant_initializer(
                                                               conv_block_weights_ini['batch_normalization'][
                                                                   'moving_var']))

                bias = _build_bias(filters,
                                   filter_ini=tf.constant_initializer(conv_block_weights_ini['convolutional_bias']))
                out_tensor = tf.nn.bias_add(out_tensor, bias)
        else:

            out_tensor = _build_conv(inp_tensor, pad_val, filter_size, filters, stride)
            bias = _build_bias(filters)
            out_tensor = tf.nn.bias_add(out_tensor, bias)
        if activate == 'leaky':
            out_tensor = tf.nn.leaky_relu(out_tensor, alpha=0.1)
    return out_tensor


def _max_pooling(inp_tensor, max_pool_block):
    size = int(max_pool_block['size'])
    stride = int(max_pool_block['stride'])
    return tf.nn.max_pool(inp_tensor,
                          ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID')


def _reorg(input_tensor, reorg_block):
    stride = int(reorg_block['stride'])
    return tf.extract_image_patches(
        input_tensor,
        [1, stride, stride, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        'VALID')


def _route(tensor_out_list, route_block):
    num_tensors = len(tensor_out_list)
    route_list = route_block.values()
    route_list = list(map(lambda route: int(route), route_list))
    route_list = [route + num_tensors for route in route_list]
    tensor_list = []
    for tensor_selection in route_list:
        tensor_list.append(tensor_out_list[int(tensor_selection)])
    return tf.concat(tensor_list, axis=3)


class YoloV2:
    def __init__(self, config_path, weights_path=None, byte_file_start=None, speak_net=False):
        # currently only process .weights file
        self.weights_path = weights_path
        self.weights_file = None
        self.speak_net = speak_net

        if self.weights_path is not None:
            if self.weights_path.endswith('.weights') is False:
                raise ValueError('The file extension is not .weights % ext' % weights_path.split('.')[-1])
            else:
                if byte_file_start is None:
                    raise ValueError('byte_file_start should be specify')
                self.weights_file = open(weights_path, 'rb')
                self.byte_count = byte_file_start
                if byte_file_start != 0:
                    self.weights_file.read(byte_file_start)

        self.config = parse_config.parse_cfg(config_path)
        self.out_tensor_list = []

    def inference(self, input_tensor):
        out_tensor = None
        prev_layer_size = []
        byte_count = 0
        if self.weights_file is not None:
            byte_count = self.byte_count
        for key, layer_block_values in self.config.items():
            # key = layer_block.keys()[0]
            # layer_block = {key: layer_block_values}
            if 'input' == key:
                prev_layer_size.append(layer_block_values['channels'])
                continue
            elif 'convolutional' in key:
                if self.weights_file is not None:
                    block_weights, tmp_byte_count = _get_weights_from_bytes(self.weights_file, layer_block_values,
                                                                            prev_layer_size[-1])
                    byte_count += tmp_byte_count
                    out_tensor = _conv(input_tensor, key, layer_block_values, conv_block_weights_ini=block_weights)
                else:
                    out_tensor = _conv(input_tensor, key, layer_block_values)
                prev_layer_size.append(layer_block_values['filters'])
            elif 'maxpool' in key:
                out_tensor = _max_pooling(input_tensor, layer_block_values)
            elif 'reorg' in key:
                out_tensor = _reorg(input_tensor, layer_block_values)
                prev_layer_size.append(out_tensor.get_shape().as_list()[3])
            elif 'route' in key:
                out_tensor = _route(self.out_tensor_list, layer_block_values)
                prev_layer_size.append(out_tensor.get_shape().as_list()[3])
            else:
                raise ValueError('No such layer')
            input_tensor = out_tensor
            if self.speak_net:
                self._print_net(key, out_tensor.get_shape().as_list())
            self.out_tensor_list.append(out_tensor)

        if self.weights_path is not None:
            remaining_bytes = len(self.weights_file.read())
            if remaining_bytes != 0:
                total_bytes = remaining_bytes + byte_count
                raise ValueError('File error found % but use %' % total_bytes % byte_count)
            self.weights_file.close()

        return out_tensor

    def _print_net(self, key, out_size):
        sentence = key + ': ' + json.dumps(self.config[key])
        out_size[0] = '?'
        sentence += ' out size: ' + json.dumps(out_size)
        print(sentence)
