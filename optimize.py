from model import VAE
from optimizer import Optimizer

discrete_parameters = {"intermediate_dim1": (50, 200), "intermediate_dim2": (20, 100), "intermediate_dim3": (10, 50),
                       "intermediate_dim4": (5, 20), "latent_dim": (3, 15)}
continuous_parameters = {"lr": (0.0001, 0.01)}
categorical_parameters = {}

optimizer = Optimizer(model_type_to_be_optimized=VAE, discrete_parameters=discrete_parameters,
                      continuous_parameters=continuous_parameters,
                      categorical_parameters=categorical_parameters)

optimizer.optimize()
