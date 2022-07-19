from core.model import VAEHandler
from utils.preprocess import preprocess

# 1. Data loading/preprocessing

# The preprocess function reads the data and performs preprocessing and encoding for the values of energy,
# angle and geometry
energies_train, cond_e_train, cond_angle_train, cond_geo_train = preprocess()

# 2. Model architecture
vae = VAEHandler()

# 3. Model training
histories = vae.train(energies_train,
                      cond_e_train,
                      cond_angle_train,
                      cond_geo_train
                      )

# Note : One history object can be used to plot the loss evaluation as function of the epochs. Remember that the
# function returns a list of those objects. Each of them represents a different fold of cross validation.
