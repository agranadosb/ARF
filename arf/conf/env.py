import os
from typing import Tuple, List, Union, IO

import yaml
from dotenv import dotenv_values

from arf.constants import RESNET_BLOCKS_ENV_VAR


def parse_blocks(blocks: Union[str, IO]) -> List[Tuple[int, int, int]]:
    parsed_yaml = yaml.safe_load(blocks)
    
    if not isinstance(parsed_yaml, dict):
        raise ValueError("Invalid YAML")
    
    block_list = parsed_yaml.get("blocks", False)
    
    if not block_list:
        raise ValueError("No blocks found in config file")
    
    return [tuple(block) for block in block_list]


environment_variables: dict[str, str] = {**dotenv_values()}

resnet_blocks_file = environment_variables.get(RESNET_BLOCKS_ENV_VAR, None)
if resnet_blocks_file is not None:
    if not os.path.isfile(resnet_blocks_file):
        raise ValueError(f"{RESNET_BLOCKS_ENV_VAR} is not a file")
    
    with open(resnet_blocks_file, "r") as f:
        RESNET_BLOCKS = parse_blocks(f)
