from typing import Tuple, Union

from torch import Tensor

TEXT_LABEL = 'text'
INDEX_LABEL = 'index'
ONE_HOT_LABEL = 'one hot'

LABEL_TYPE_OPTIONS = [TEXT_LABEL, INDEX_LABEL, ONE_HOT_LABEL]

PAIR_TYPE = Tuple[Tensor, Union[str, int, Tensor]]

# Enviorment variables
RESNET_BLOCKS_ENV_VAR = 'RESNET_BLOCKS'
RESNET_BATCH_SIZE_VAR = 'RESNET_BATCH_SIZE'
RESNET_EPOCHS_VAR = 'RESNET_EPOCHS'
RESNET_IMAGE_SIZE_VAR = 'RESNET_IMAGE_SIZE'

TRAINING_DATA_VAR = 'TRAINING_DATA'
VALIDATION_DATA_VAR = 'VALIDATION_DATA'
TEST_DATA_VAR = 'TEST_DATA'

ENV_VARIABLES = {
    RESNET_BLOCKS_ENV_VAR: """
        Yaml file with the number of blocks for resnet network. This file should be like this:
        
        blocks:
          -
            - 1     # repetitions of this block
            - 3     # kernel_size
            - 64    # out_channels
          -
            - 1     # repetitions of this block
            - 3     # kernel_size
            - 128   # out_channels
        
        At the start of each block (starting from second block) a stride 2 will be applied.
    """,
    RESNET_BATCH_SIZE_VAR: "Batch size for resnet training (32 by default)",
    RESNET_EPOCHS_VAR: "Epochs for resnet training (100 by default)",
    RESNET_IMAGE_SIZE_VAR: "Image size for resnet training (512 by default)",
    TRAINING_DATA_VAR: (
        "Folder where the training data is stored. This folder should divided the data in one folder per class."
    ),
    VALIDATION_DATA_VAR: (
        "Folder where the validation data is stored. This folder should divided the data in one folder per class."
    ),
    TEST_DATA_VAR: (
        "Folder where the test data is stored. This folder should divided the data in one folder per class."
    ),
}
