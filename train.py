"""
** train **
performs the training
"""

# 1. Data loading/preprocessing
from preprocess import preprocess
# The preprocess function reads the data and performs preprocessing and encoding for the values of energy, angle and geometry
energies_Train,condE_Train,condAngle_Train,condGeo_Train = preprocess()

# 2. Model architecture
from instantiate_model import *
vae = instantiate()

# 3. Model training
history = vae.train(energies_Train,
        condE_Train,
        condAngle_Train,
        condGeo_Train
        )

# Note : the history object can be used to plot the loss evaluation as function of the epochs.



