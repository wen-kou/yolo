import unittest
import os

from yolov2.utils import parse_config


class MyTestCase(unittest.TestCase):
    def test_parse_cfg(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.dirname(current_dir) + '/assets/cfg/yolo.cfg'
        net_config, bbox_config = parse_config.parse_cfg(config_path)
        self.assertEqual(32, len(net_config))
        self.assertEqual(32, bbox_config['input_size'][0]//bbox_config['out_size'][0])


if __name__ == '__main__':
    unittest.main()
