"""
** instantiate_model **
instantiate a VAE model and define all the parameters
"""

from constants import BATCH_SIZE, ORIGINAL_DIM, INTERMEDIATE_DIM1, INTERMEDIATE_DIM2, INTERMEDIATE_DIM3, \
    INTERMEDIATE_DIM4, LATENT_DIM, EPOCHS, LR, ACTIVATION, OUT_ACTIVATION, VALIDATION_SPLIT, \
    OPTIMIZER, KERNEL_INITIALIZER, BIAS_INITIALIZER, EARLY_STOP, CHECKPOINT_DIR
from model import VAE


def instantiate():
    vae = VAE(batch_size=BATCH_SIZE,
              original_dim=ORIGINAL_DIM,
              intermediate_dim1=INTERMEDIATE_DIM1,
              intermediate_dim2=INTERMEDIATE_DIM2,
              intermediate_dim3=INTERMEDIATE_DIM3,
              intermediate_dim4=INTERMEDIATE_DIM4,
              latent_dim=LATENT_DIM,
              epochs=EPOCHS,
              lr=LR,
              activation=ACTIVATION,
              out_activation=OUT_ACTIVATION,
              validation_split=VALIDATION_SPLIT,
              optimizer=OPTIMIZER,
              kernel_initializer=KERNEL_INITIALIZER,
              bias_initializer=BIAS_INITIALIZER,
              early_stop=EARLY_STOP,
              checkpoint_dir=CHECKPOINT_DIR
              )
    return vae
