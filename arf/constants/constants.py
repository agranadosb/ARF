RESNET_BLOCKS_ENV_VAR = 'RESNET_BLOCKS'

ENV_VARIABLES = {RESNET_BLOCKS_ENV_VAR: """
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
"""}
