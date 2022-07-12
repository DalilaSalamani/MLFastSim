from hyperparameter_tuner import HyperparameterTuner
from model import OptimizerType

discrete_parameters = {"intermediate_dim1": (50, 200), "intermediate_dim2": (20, 100), "intermediate_dim3": (10, 50),
                       "intermediate_dim4": (5, 20), "latent_dim": (3, 15)}
continuous_parameters = {"learning_rate": (0.0001, 0.01)}
categorical_parameters = {"optimizer": [OptimizerType.SGD, OptimizerType.ADAM]}

optimizer = HyperparameterTuner(discrete_parameters=discrete_parameters,
                                continuous_parameters=continuous_parameters,
                                categorical_parameters=categorical_parameters)

optimizer.tune()
