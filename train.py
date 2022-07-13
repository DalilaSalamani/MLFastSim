from preprocess import preprocess
from core.model import VAE

# 1. Data loading/preprocessing

# The preprocess function reads the data and performs preprocessing and encoding for the values of energy,
# angle and geometry
energies_train, cond_e_train, cond_angle_train, cond_geo_train = preprocess()

# 2. Model architecture
vae = VAE()

# 3. Model training
history = vae.train(energies_train,
                    cond_e_train,
                    cond_angle_train,
                    cond_geo_train
                    )

# Note : the history object can be used to plot the loss evaluation as function of the epochs.
