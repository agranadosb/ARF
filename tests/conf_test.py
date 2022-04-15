from unittest import TestCase
from unittest.mock import patch

from arf.constants import RESNET_BLOCKS_ENV_VAR

with patch("dotenv.dotenv_values", lambda: {RESNET_BLOCKS_ENV_VAR: "tests/static/blocks_yaml.yml"}):
    from arf.conf.env import parse_blocks
    from arf.conf.env import RESNET_BLOCKS


class TestEnv(TestCase):
    def setUp(self):
        super().setUp()
    
    def test_parse_blocks(self):
        blocks = """
            blocks:
                -
                    - 1
                    - 3
                    - 64
                -
                    - 1
                    - 3
                    - 128
        """
        correct_blocks = [(1, 3, 64), (1, 3, 128)]
        
        result_blocks = parse_blocks(blocks)
        
        self.assertEqual(result_blocks, correct_blocks)
    
    def test_parse_blocks_empty(self):
        blocks = """
            blocks:
        """
        with self.assertRaises(ValueError):
            parse_blocks(blocks)
    
    def test_parse_blocks_incorrect_yaml(self):
        blocks = """
            blocks=[12, w]
        """
        with self.assertRaises(ValueError):
            parse_blocks(blocks)
    
    def test_parse_blocks_yml_file(self):
        blocks_yml_file = "tests/static/blocks_yaml.yml"
        correct_blocks = [(1, 3, 64), (1, 3, 128)]
        
        with open(blocks_yml_file, 'r') as fr:
            result_blocks = parse_blocks(fr)
        
        self.assertEqual(result_blocks, correct_blocks)
    
    def test_RESNET_BLOCKS_from_a_file(self):
        correct_blocks = [(1, 3, 64), (1, 3, 128)]
        
        self.assertEqual(RESNET_BLOCKS, correct_blocks)
