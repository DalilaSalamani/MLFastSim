from typing import Tuple, Dict, Any, List

import optuna
import tensorflow as tf
from optuna import Trial
from optuna.trial import TrialState

from constants import LR, BATCH_SIZE, ORIGINAL_DIM, INTERMEDIATE_DIM1, INTERMEDIATE_DIM2, INTERMEDIATE_DIM3, \
    INTERMEDIATE_DIM4, EPOCHS, ACTIVATION, OUT_ACTIVATION, VALIDATION_SPLIT, CHECKPOINT_DIR, OPTIMIZER, \
    KERNEL_INITIALIZER, BIAS_INITIALIZER, N_TRIALS, LATENT_DIM, SAVE_FREQ
from model import VAE
from preprocess import preprocess


class Optimizer:
    """ Optimizer which looks for the best hyperparameters of a Variational Autoencoder specified in model.py.

    Attributes:
        _discrete_parameters: A dictionary of hyperparameters taking discrete values in the range [low, high].
        _continuous_parameters: A dictionary of hyperparameters taking continuous values in the range [low, high].
        _categorical_parameters: A dictionary of hyperparameters taking values specified by the list of them.

    """

    def __init__(self, discrete_parameters: Dict[str, Tuple[int, int]],
                 continuous_parameters: Dict[str, Tuple[float, float]], categorical_parameters: Dict[str, List[Any]]):
        self._discrete_parameters = discrete_parameters
        self._continuous_parameters = continuous_parameters
        self._categorical_parameters = categorical_parameters
        self._energies_train, self._cond_e_train, self._cond_angle_train, self._cond_geo_train = preprocess()

    def _create_model(self, trial: Trial) -> VAE:
        """For a given trail builds the model.

        Optuna suggests parameters like dimensions of particular layers of the model, learning rate, optimizer, etc.

        Args:
            trial: Optuna's trial

        Returns:
            Variational Autoencoder (VAE)
        """
        # Discrete parameters
        if "original_dim" in self._discrete_parameters.keys():
            original_dim = trial.suggest_int("original_dim", self._discrete_parameters["original_dim"][0],
                                             self._discrete_parameters["original_dim"][1])
        else:
            original_dim = ORIGINAL_DIM
        if "intermediate_dim1" in self._discrete_parameters.keys():
            intermediate_dim1 = trial.suggest_int("intermediate_dim1",
                                                  self._discrete_parameters["intermediate_dim1"][0],
                                                  self._discrete_parameters["intermediate_dim1"][1])
        else:
            intermediate_dim1 = INTERMEDIATE_DIM1
        if "intermediate_dim2" in self._discrete_parameters.keys():
            intermediate_dim2 = trial.suggest_int("intermediate_dim2",
                                                  self._discrete_parameters["intermediate_dim2"][0],
                                                  self._discrete_parameters["intermediate_dim2"][1])
        else:
            intermediate_dim2 = INTERMEDIATE_DIM2
        if "intermediate_dim3" in self._discrete_parameters.keys():
            intermediate_dim3 = trial.suggest_int("intermediate_dim3",
                                                  self._discrete_parameters["intermediate_dim3"][0],
                                                  self._discrete_parameters["intermediate_dim3"][1])
        else:
            intermediate_dim3 = INTERMEDIATE_DIM3
        if "intermediate_dim4" in self._discrete_parameters.keys():
            intermediate_dim4 = trial.suggest_int("intermediate_dim4",
                                                  self._discrete_parameters["intermediate_dim4"][0],
                                                  self._discrete_parameters["intermediate_dim4"][1])
        else:
            intermediate_dim4 = INTERMEDIATE_DIM4
        if "latent_dim" in self._discrete_parameters.keys():
            latent_dim = trial.suggest_int("latent_dim",
                                           self._discrete_parameters["latent_dim"][0],
                                           self._discrete_parameters["latent_dim"][1])
        else:
            latent_dim = LATENT_DIM

        # Continuous parameters
        if "lr" in self._continuous_parameters.keys():
            lr = trial.suggest_float("lr", low=self._continuous_parameters["lr"][0],
                                     high=self._continuous_parameters["lr"][1])
        else:
            lr = LR

        # Categorical parameters
        if "activation" in self._categorical_parameters.keys():
            activation = trial.suggest_categorical("activation", self._categorical_parameters["activation"])
        else:
            activation = ACTIVATION
        if "out_activation" in self._categorical_parameters.keys():
            out_activation = trial.suggest_categorical("out_activation", self._categorical_parameters["out_activation"])
        else:
            out_activation = OUT_ACTIVATION
        if "optimizer" in self._categorical_parameters.keys():
            optimizer = trial.suggest_categorical("optimizer", self._categorical_parameters["optimizer"])
        else:
            optimizer = OPTIMIZER
        if "kernel_initializer" in self._categorical_parameters.keys():
            kernel_initializer = trial.suggest_categorical("kernel_initializer",
                                                           self._categorical_parameters["kernel_initializer"])
        else:
            kernel_initializer = KERNEL_INITIALIZER
        if "bias_initializer" in self._categorical_parameters.keys():
            bias_initializer = trial.suggest_categorical("bias_initializer",
                                                         self._categorical_parameters["bias_initializer"])
        else:
            bias_initializer = BIAS_INITIALIZER

        model = VAE(batch_size=BATCH_SIZE,
                    original_dim=original_dim,
                    intermediate_dim1=intermediate_dim1,
                    intermediate_dim2=intermediate_dim2,
                    intermediate_dim3=intermediate_dim3,
                    intermediate_dim4=intermediate_dim4,
                    latent_dim=latent_dim,
                    epochs=EPOCHS,
                    lr=lr,
                    activation=activation,
                    out_activation=out_activation,
                    validation_split=VALIDATION_SPLIT,
                    optimizer=optimizer,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    early_stop=True,
                    checkpoint_dir=CHECKPOINT_DIR,
                    save_freq=SAVE_FREQ)
        return model

    def _objective(self, trial: Trial) -> float:
        """For a given trial trains the model and returns validation loss.

        Args:
            trial: Optuna's trial

        Returns:
            One float numer which is a validation loss calculated on 5% unseen before elements of the dataset.
        """
        tf.keras.backend.clear_session()

        # Generate the trial model.
        model = self._create_model(trial)

        # Train the model.
        verbose = True
        history = model.train(self._energies_train,
                              self._cond_e_train,
                              self._cond_angle_train,
                              self._cond_geo_train,
                              verbose
                              )

        # Return validation loss (currently it is treated as an objective goal).
        validation_loss_history = history.history["val_loss"]
        final_validation_loss = validation_loss_history[-1]
        return final_validation_loss

    def optimize(self) -> None:
        """Main optimization function.

        It creates a study, optimize model and prints detailed information
        about the best trial (value of the objective function and adjusted parameters).
        """
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(self._objective, n_trials=N_TRIALS)
        pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
