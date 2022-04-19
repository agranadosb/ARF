import os
from typing import Tuple, List, Union, IO

import yaml
from dotenv import dotenv_values

from arf.constants import RESNET_BLOCKS_ENV_VAR, RESNET_BATCH_SIZE_VAR, TEST_DATA_VAR, VALIDATION_DATA_VAR, \
    TRAINING_DATA_VAR


def parse_blocks(blocks: Union[str, IO]) -> List[Tuple[int, int, int]]:
    """This function gets a yml string like or a yml file representing the list
    of resnet blocks and return the parsed blocks. The definition of this file
    should be like this:

    ```yaml
    blocks:
      -
        - 1     # repetitions of this block
        - 3     # kernel_size
        - 64    # out_channels
      -
        - 1     # repetitions of this block
        - 3     # kernel_size
        - 128   # out_channels
    ```
    
    Parameters
    ----------
    blocks : Union[str, IO]
        A yml string or a yml file representing the list of resnet blocks.

    Returns
    -------
    A list of tuples representing the blocks.
    """
    parsed_yaml = yaml.safe_load(blocks)
    
    if not isinstance(parsed_yaml, dict):
        raise ValueError("Invalid YAML")
    
    block_list = parsed_yaml.get("blocks", [])
    
    if not block_list:
        raise ValueError("No blocks found in config file")
    
    return [tuple(block) for block in block_list]


environment_variables: dict[str, str] = {**dotenv_values()}

# Resnet
resnet_blocks_file = environment_variables.get(RESNET_BLOCKS_ENV_VAR, None)
RESNET_BLOCKS = None
if resnet_blocks_file is not None:
    if not os.path.isfile(resnet_blocks_file):
        raise ValueError(f"{resnet_blocks_file} is not a file")

    with open(resnet_blocks_file, "r") as f:
        RESNET_BLOCKS = parse_blocks(f)

RESNET_BATCH_SIZE = int(environment_variables.get(RESNET_BATCH_SIZE_VAR, "32"))

# Data Folders
TRAINING_DATA = environment_variables.get(TRAINING_DATA_VAR, None)
VALIDATION_DATA = environment_variables.get(VALIDATION_DATA_VAR, None)
TEST_DATA = environment_variables.get(TEST_DATA_VAR, None)
