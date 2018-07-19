import configparser
import io

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
    layer_block = dict()
    for section in config_parser.sections():
        if section.startswith('convolutional'):
            filters = int(config_parser[section]['filters'])
            size = int(config_parser[section]['size'])
            stride = int(config_parser[section]['stride'])
            pad = 'SAME' if int(config_parser[section]['pad']) == 1 else 'VALID'
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
            layer_block.update({section: tmp_dict})
        elif section.startswith('maxpool'):
            size = int(config_parser[section]['size'])
            stride = int(config_parser[section]['stride'])
            tmp_dict = {'size': size,
                        'stride': stride}
            layer_block.update({section: tmp_dict})
        elif section.startswith('route'):
            ids = [int(i) for i in config_parser[section]['layers'].split(',')]
            # layers = [config_parser[i] for i in ids]
            tmp_dict = dict()
            for i, layer in enumerate(ids):
                tmp_dict.update({'layer' + str(i): layer})
            layer_block.update({section: tmp_dict})
        elif section.startswith('reorg'):
            block_size = int(config_parser[section]['stride'])
            tmp_dict = {'stride': block_size}
            layer_block.update({section: tmp_dict})
        elif section.startswith('net'):
            height = config_parser[section]['height']
            width = config_parser[section]['width']
            channels = config_parser[section]['channels']
            tmp_dict = {
                'height': height,
                'width': width,
                'channels': int(channels)
            }
            layer_block.update({'input': tmp_dict})
    return layer_block
