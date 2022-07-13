from tensorflow.python.keras.layers import LeakyReLU

from model import OptimizerType

"""
Experiment constants.
"""
# Number of calorimeter layers (z-axis segmentation).
N_CELLS_Z = 45
# Segmentation in the r,phi direction.
N_CELLS_R = 18
N_CELLS_PHI = 50
# Cell size in the r and z directions 
SIZE_R = 2.325
SIZE_Z = 3.4

# Minimum and maximum primary particle energy to consider for training in GeV units.
MIN_ENERGY = 1
MAX_ENERGY = 1024
# Minimum and maximum primary particle angle to consider for training in degrees units.
MIN_ANGLE = 50
MAX_ANGLE = 90

"""
Directories.
"""
# Directory to load the full simulation dataset.
INIT_DIR = "./dataset/"
# Directory to save VAE checkpoints
CHECKPOINT_DIR = "./checkpoint/"
# Directory to save model after conversion to a format that can be used in C++.
CONV_DIR = "./conversion/"
# Directory to save validation plots.
VALID_DIR = "./validation/"
# Directory to save VAE generated showers.
GEN_DIR = "./generation/"

"""
Model default parameters.
"""
BATCH_SIZE = 100
# Total number of readout cells (represents the number of nodes in the input/output layers of the model).
ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI
INTERMEDIATE_DIM1 = 100
INTERMEDIATE_DIM2 = 50
INTERMEDIATE_DIM3 = 20
INTERMEDIATE_DIM4 = 10 + 4
LATENT_DIM = 10
EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_FREQ = 100
ACTIVATION = LeakyReLU()
OUT_ACTIVATION = "sigmoid"
VALIDATION_SPLIT = 0.05
OPTIMIZER_TYPE = OptimizerType.ADAM
KERNEL_INITIALIZER = "RandomNormal"
BIAS_INITIALIZER = "Zeros"
EARLY_STOP = False

"""
Optimizer parameters.
"""
N_TRIALS = 5
# Maximum allowed memory on one of the GPUs (in GB)
MAX_GPU_MEMORY_ALLOCATION = 8
# ID of GPU used in a process
GPU_ID = 0
