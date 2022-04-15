from unittest import TestCase
from unittest.mock import patch

with patch('arf.constants.RESNET_BLOCKS_ENV_VAR', "NotExistsThisKey"):
    from arf.conf.env import parse_blocks

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
