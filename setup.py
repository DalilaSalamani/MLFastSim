"""
** setup **
creates necessary folders
"""

import os

from configure import Configure

variables = Configure()

for folder in [variables.init_dir,  # Directory to load the full simulation dataset
               variables.checkpoint_dir,  # Directory to save VAE checkpoints
               variables.conv_dir,  # Directory to save model after conversion to a format that can be used in C++
               variables.valid_dir,  # Directory to save validation plots
               variables.gen_dir,  # Directory to save VAE generated showers
               ]:
    os.system("mkdir %s" % folder)
