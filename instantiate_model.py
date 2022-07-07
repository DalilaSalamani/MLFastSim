"""
** instantiate_model **
instantiate a VAE model and define all the parameters
"""

import tensorflow

import model
from configure import Configure


# instantiate function returns an instance of the VAE model
def instantiate():
    # Get list of common variables
    variables = Configure()
    vae = model.VAE(batch_size=100,
                    original_dim=variables.original_dim,
                    intermediate_dim1=100,
                    intermediate_dim2=50,
                    intermediate_dim3=20,
                    intermediate_dim4=10 + 4,
                    latent_dim=10,
                    epsilon_std=1.,
                    mu=0,
                    epochs=10000,
                    lr=0.001,
                    activ=tensorflow.keras.layers.LeakyReLU(),
                    outActiv="sigmoid",
                    validation_split=0.05,
                    wReco=variables.original_dim,
                    wkl=0.5,
                    optimizer=tensorflow.keras.optimizers.Adam(),
                    ki="RandomNormal",
                    bi="Zeros",
                    earlyStop=False,
                    checkpoint_dir=variables.checkpoint_dir
                    )
    return vae
