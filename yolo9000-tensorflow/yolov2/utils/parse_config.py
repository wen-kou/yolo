import configparser
import io
import json

from collections import defaultdict


def _unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def parse_cfg(config_path):
    config_parser = configparser.ConfigParser()
    unique_config_file = _unique_config_sections(config_path)
    config_parser.read_file(unique_config_file)
    net_config = dict()
    bbox_config = dict()
    out_size_list = list()
    for section in config_parser.sections():
        if section.startswith('net'):
            height = config_parser[section]['height']
            width = config_parser[section]['width']
            channels = config_parser[section]['channels']
            tmp_dict = {
                'height': int(height),
                'width': int(width),
                'channels': int(channels)
            }
            out_size_list.append((int(height), int(width)))
            bbox_config.update({'input_size': (int(height), int(width))})
            net_config.update({'input': tmp_dict})
        elif section.startswith('convolutional'):
            filters = int(config_parser[section]['filters'])
            size = int(config_parser[section]['size'])
            stride = int(config_parser[section]['stride'])
            pad = 'SAME' if int(config_parser[section]['pad']) == 1 else 'VALID'
            if pad == 'VALID':
                out_size = ((out_size_list[-1][0] + size + 2 * int(config_parser[section]['pad']))/stride + 1,
                            (out_size_list[-1][1] + size + 2 * int(config_parser[section]['pad'])) / stride + 1)
                out_size_list.append(out_size)
            else:
                out_size_list.append(out_size_list[-1])
            activation = config_parser[section]['activation']
            batch_normalize = 'batch_normalize' in config_parser[section]
            tmp_dict = {
                'filters': filters,
                'size': size,
                'stride': stride,
                'pad': pad,
                'activation': activation,
                'batch_normalize': batch_normalize
            }
            net_config.update({section: tmp_dict})
        elif section.startswith('maxpool'):
            size = int(config_parser[section]['size'])
            stride = int(config_parser[section]['stride'])
            tmp_dict = {'size': size,
                        'stride': stride}
            out_size = ((out_size_list[-1][0] - size) / stride + 1,
                        (out_size_list[-1][1] - size) / stride + 1)
            out_size_list.append(out_size)
            net_config.update({section: tmp_dict})
        elif section.startswith('route'):
            ids = [int(i) for i in config_parser[section]['layers'].split(',')]
            # layers = [config_parser[i] for i in ids]
            tmp_dict = dict()
            num_current_layers = len(out_size_list)
            tmp = []
            for i, layer in enumerate(ids):
                tmp_dict.update({'layer' + str(i): layer})
                out_size = out_size_list[num_current_layers + layer]
                tmp.append(out_size)
            tmp = list(set(tmp))
            if len(tmp) > 1:
                raise ValueError('layer %  can not be concatenated' % ids)
            out_size_list.append(tmp[0])
            net_config.update({section: tmp_dict})
        elif section.startswith('reorg'):
            block_size = int(config_parser[section]['stride'])
            tmp_dict = {'stride': block_size}
            if (out_size_list[-1][0] % block_size != 0) | (out_size_list[-1][1] % block_size != 0):
                raise ValueError('layer can not be re-orgnized')
            out_size = (out_size_list[-1][0] / block_size,
                        out_size_list[-1][1] / block_size)
            out_size_list.append(out_size)
            net_config.update({section: tmp_dict})
        elif section.startswith('region'):
            anchors = config_parser[section]['anchors'].split(',')
            tmp = []
            for i in range(0, len(anchors),2):
                tmp.append([float(anchors[i]), float(anchors[i + 1])])

            n_classes = int(config_parser[section]['classes'])
            n_coords = int(config_parser[section]['coords'])
            thresh = float(config_parser[section]['thresh'])
            bbox_config = {'anchors': tmp,
                           'classes': n_classes,
                           'coords': n_coords,
                           'out_size': out_size_list[-1],
                           'thresh': thresh,
                           'input_size': out_size_list[0],
                           'n_classes': int(config_parser[section]['classes'])}
    return net_config, bbox_config
