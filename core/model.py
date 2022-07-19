from typing import List, Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History, Callback
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Layer, concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model

from core.constants import ORIGINAL_DIM, LATENT_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, ACTIVATION, OUT_ACTIVATION, \
    OPTIMIZER_TYPE, KERNEL_INITIALIZER, CHECKPOINT_DIR, EARLY_STOP, BIAS_INITIALIZER, PERIOD, \
    INTERMEDIATE_DIMS, SAVE_MODEL, SAVE_BEST, PATIENCE, MIN_DELTA, BEST_MODEL_FILENAME, NUMBER_OF_K_FOLD_SPLITS, \
    VALIDATION_SPLIT
from utils.optimizer import OptimizerFactory, OptimizerType


class _Sampling(Layer):
    """ Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
    distribution.

    The sampled vector z is given by sampled_z = mean + std * epsilon
    """

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        z_sigma = tf.math.exp(0.5 * z_log_var)
        epsilon = tf.random.normal(tf.shape(z_sigma))
        return z_mean + z_sigma * epsilon


class _Reparametrize(Layer):
    """
    Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
    distribution.

    The sampled vector z is given by sampled_z = mean + std * epsilon
    """

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        z_sigma = tf.math.exp(0.5 * z_log_var)
        epsilon = tf.random.normal(tf.shape(z_sigma))
        return z_mean + z_sigma * epsilon


class VAE(Model):
    def get_config(self):
        config = super().get_config()
        config["encoder"] = self.encoder
        config["decoder"] = self.decoder
        return config

    def call(self, inputs, training=None, mask=None):
        _, e_input, angle_input, geo_input = inputs
        z, _, _ = self.encoder(inputs)
        return self.decoder([z, e_input, angle_input, geo_input])

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._set_inputs(inputs=self.encoder.inputs, outputs=self(self.encoder.inputs))
        self.total_loss_tracker = Mean(name="total_loss")
        self.val_total_loss_tracker = Mean(name="val_total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.val_total_loss_tracker]

    def _perform_step(self, data: Any) -> Tuple[float, float, float]:
        # Unpack data.
        ([x_input, e_input, angle_input, geo_input],) = data
        # Encode data and get new probability distribution with a vector z sampled from it.
        z_mean, z_log_var, z = self.encoder([x_input, e_input, angle_input, geo_input])
        # Reconstruct the original data.
        reconstruction = self.decoder([z, e_input, angle_input, geo_input])

        # Reshape data.
        x_input = tf.expand_dims(x_input, -1)
        reconstruction = tf.expand_dims(reconstruction, -1)

        # Calculate reconstruction loss.
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(binary_crossentropy(x_input, reconstruction), axis=1))

        # Calculate Kullback-Leibler divergence.
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # Calculated weighted total loss (ORIGINAL_DIM is a weight).
        total_loss = ORIGINAL_DIM * reconstruction_loss + kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data: Any) -> Dict[str, object]:
        with tf.GradientTape() as tape:
            # Perform step, backpropagate it through the network and update the tracker.
            total_loss, reconstruction_loss, kl_loss = self._perform_step(data)

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)

        return {"total_loss": self.total_loss_tracker.result()}

    def test_step(self, data: Any) -> Dict[str, object]:
        # Perform step and update the tracker (no backpropagation).
        val_total_loss, val_reconstruction_loss, val_kl_loss = self._perform_step(data)

        self.val_total_loss_tracker.update_state(val_total_loss)

        return {"total_loss": self.val_total_loss_tracker.result()}


def _get_train_and_val_data(dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                            train_indexes: np.array, validation_indexes: np.array) -> List[List[np.array]]:
    """
    Splits data into train and validation set based on given lists of indexes.

    """
    train_dataset = dataset[train_indexes, :]
    train_e_cond = e_cond[train_indexes]
    train_angle_cond = angle_cond[train_indexes]
    train_geo_cond = geo_cond[train_indexes, :]

    val_dataset = dataset[validation_indexes, :]
    val_e_cond = e_cond[validation_indexes]
    val_angle_cond = angle_cond[validation_indexes]
    val_geo_cond = geo_cond[validation_indexes, :]

    train_data = [train_dataset, train_e_cond, train_angle_cond, train_geo_cond]
    val_data = [val_dataset, val_e_cond, val_angle_cond, val_geo_cond]

    return [train_data, val_data]


class VAEHandler:
    """
    Class to handle building and training VAE models.
    """

    def __init__(self, original_dim: int = ORIGINAL_DIM, latent_dim: int = LATENT_DIM, batch_size: int = BATCH_SIZE,
                 intermediate_dims: List[int] = INTERMEDIATE_DIMS,
                 learning_rate: float = LEARNING_RATE, epochs: int = EPOCHS, activation: str = ACTIVATION,
                 out_activation: str = OUT_ACTIVATION, number_of_k_fold_splits: float = NUMBER_OF_K_FOLD_SPLITS,
                 optimizer_type: OptimizerType = OPTIMIZER_TYPE, kernel_initializer: str = KERNEL_INITIALIZER,
                 bias_initializer: str = BIAS_INITIALIZER, checkpoint_dir: str = CHECKPOINT_DIR,
                 early_stop: bool = EARLY_STOP, save_model: bool = SAVE_MODEL, save_best: bool = SAVE_BEST,
                 period: int = PERIOD, patience: int = PATIENCE, min_delta: float = MIN_DELTA,
                 best_model_filename: str = BEST_MODEL_FILENAME, validation_split: float = VALIDATION_SPLIT):
        self._validation_split = validation_split
        self._best_model_filename = best_model_filename
        self._min_delta = min_delta
        self._patience = patience
        self._save_best = save_best
        self._save_model = save_model
        self._original_dim = original_dim
        self.latent_dim = latent_dim
        self._intermediate_dims = intermediate_dims
        self._batch_size = batch_size
        self._epochs = epochs
        self._activation = activation
        self._out_activation = out_activation
        self._number_of_k_fold_splits = number_of_k_fold_splits
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._checkpoint_dir = checkpoint_dir
        self._early_stop = early_stop
        self._period = period

        # Build encoder and decoder.
        encoder = self._build_encoder()
        decoder = self._build_decoder()

        # Build VAE.
        self.model = VAE(encoder, decoder)

        # Manufacture an optimizer and compile model with.
        optimizer = OptimizerFactory.create_optimizer(optimizer_type, learning_rate)
        self.model.compile(optimizer=optimizer,
                           metrics=[self.model.total_loss_tracker, self.model.val_total_loss_tracker])

    def _prepare_input_layers(self, for_encoder: bool) -> Tuple[Input, Input, Input, Input]:
        """
        Create four Input layers. Each of them is responsible to take respectively: batch of showers/batch of latent
        vectors, batch of energies, batch of angles, batch of geometries.

        Args:
            for_encoder: Boolean which decides whether an input is full dimensional shower or a latent vector.

        Returns:
            Tuple of four Input layers.

        """
        x_input = Input(shape=self._original_dim) if for_encoder else Input(shape=self.latent_dim)
        e_input = Input(shape=(1,))
        angle_input = Input(shape=(1,))
        geo_input = Input(shape=(2,))
        return x_input, e_input, angle_input, geo_input

    def _build_encoder(self) -> Model:
        """ Based on a list of intermediate dimensions, activation function and initializers for kernel and bias builds
        the encoder.

        Returns:
             Encoder is returned as a keras.Model.

        """

        # Prepare input layer.
        x_input, e_input, angle_input, geo_input = self._prepare_input_layers(for_encoder=True)
        x = concatenate([x_input, e_input, angle_input, geo_input])
        # Construct hidden layers (Dense and Batch Normalization).
        for intermediate_dim in self._intermediate_dims:
            x = Dense(units=intermediate_dim, activation=self._activation, kernel_initializer=self._kernel_initializer,
                      bias_initializer=self._bias_initializer)(x)
            x = BatchNormalization()(x)
        # Add Dense layer to get description of multidimensional Gaussian distribution in terms of mean
        # and log(variance).
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        # Sample a probe from the distribution.
        z = _Sampling()([z_mean, z_log_var])
        # Return model.
        return Model(inputs=[x_input, e_input, angle_input, geo_input], outputs=[z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self) -> Model:
        """ Based on a list of intermediate dimensions, activation function and initializers for kernel and bias builds
        the decoder.

        Returns:
             Decoder is returned as a keras.Model.

        """

        # Prepare input layer.
        latent_input, e_input, angle_input, geo_input = self._prepare_input_layers(for_encoder=False)
        x = concatenate([latent_input, e_input, angle_input, geo_input])
        # Construct hidden layers (Dense and Batch Normalization).
        for intermediate_dim in reversed(self._intermediate_dims):
            x = Dense(units=intermediate_dim, activation=self._activation, kernel_initializer=self._kernel_initializer,
                      bias_initializer=self._bias_initializer)(x)
            x = BatchNormalization()(x)
        # Add Dense layer to get output which shape is compatible in an input's shape.
        decoder_outputs = Dense(units=self._original_dim, activation=self._out_activation)(x)
        # Return model.
        return Model(inputs=[latent_input, e_input, angle_input, geo_input], outputs=decoder_outputs, name="decoder")

    def _manufacture_callbacks(self) -> List[Callback]:
        """
        Based on parameters set by the user, manufacture callbacks required for training.

        Returns:
            A list of `Callback` objects.

        """
        # If the early stopping flag is on then stop the training when a monitored metric (validation) has stopped
        # improving after (patience) number of epochs.
        callbacks = []
        if self._early_stop:
            callbacks.append(
                EarlyStopping(monitor="val_total_loss",
                              min_delta=self._min_delta,
                              patience=self._patience,
                              verbose=True,
                              restore_best_weights=True))
        # If the save model flag is on then save model every (self._period) number of epochs regardless of
        # performance of the model.
        if self._save_model:
            callbacks.append(ModelCheckpoint(filepath=f"{self._checkpoint_dir}VAE-{{epoch:02d}}.tf",
                                             monitor="val_total_loss",
                                             verbose=True,
                                             save_weights_only=False,
                                             mode="min",
                                             period=self._period))
        return callbacks

    def _k_fold_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         callbacks: List[Callback], verbose: bool = True) -> List[History]:
        """
        Performs K-fold cross validation training.

        Number of fold is defined by (self._number_of_k_fold_splits). Always shuffle the dataset.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            callbacks: A list of callback forwarded to the fitting function.
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A list of `History` objects.`History.history` attribute is a record of training loss values and
        metrics values at successive epochs, as well as validation loss values and validation metrics values (if
        applicable).

        """
        k_fold = KFold(n_splits=self._number_of_k_fold_splits, shuffle=True)
        histories = []

        for i, (train_indexes, validation_indexes) in enumerate(k_fold.split(dataset)):
            print(f"K-fold: {i + 1}/{self._number_of_k_fold_splits}...")
            train_data, val_data = _get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, train_indexes,
                                                           validation_indexes)

            history = self.model.fit(x=train_data,
                                     shuffle=True,
                                     epochs=EPOCHS,
                                     verbose=verbose,
                                     validation_data=(val_data, None),
                                     batch_size=self._batch_size,
                                     callbacks=callbacks
                                     )
            histories.append(history)

            if self._save_best:
                self.model.save(f"{self._checkpoint_dir}{self._best_model_filename}_{i}.tf")
                print("Best model was saved.")

        return histories

    def _single_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         callbacks: List[Callback], verbose: bool = True) -> List[History]:
        """
        Performs a single training.

        A fraction of dataset (self._validation_split) is used as a validation data.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            callbacks: A list of callback forwarded to the fitting function.
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A one-element list of `History` objects.`History.history` attribute is a record of training loss
        values and metrics values at successive epochs, as well as validation loss values and validation metrics
        values (if applicable).

        """
        dataset_size, _ = dataset.shape
        permutation = np.random.permutation(dataset_size)
        split = int(dataset_size * self._validation_split)
        train_indexes, validation_indexes = permutation[split:], permutation[:split]

        train_data, val_data = _get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, train_indexes,
                                                       validation_indexes)
        history = self.model.fit(x=train_data,
                                 shuffle=True,
                                 epochs=EPOCHS,
                                 verbose=verbose,
                                 validation_data=(val_data, None),
                                 batch_size=self._batch_size,
                                 callbacks=callbacks
                                 )
        if self._save_best:
            self.model.save(f"{self._checkpoint_dir}{self._best_model_filename}.tf")
            print("Best model was saved.")

        return [history]

    def train(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
              verbose: bool = True) -> List[History]:
        """
        For a given input data trains and validates the model.

        If the numer of K-fold splits > 1 then it runs K-fold cross validation, otherwise it runs a single training
        which uses (self._validation_split * 100) % of dataset as a validation data.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A list of `History` objects.`History.history` attribute is a record of training loss values and
        metrics values at successive epochs, as well as validation loss values and validation metrics values (if
        applicable).

        """

        callbacks = self._manufacture_callbacks()

        if self._number_of_k_fold_splits > 1:
            return self._k_fold_training(dataset, e_cond, angle_cond, geo_cond, callbacks, verbose)
        else:
            return self._single_training(dataset, e_cond, angle_cond, geo_cond, callbacks, verbose)
