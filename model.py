"""
** model **
defines the VAE model class 
"""

import numpy
from tensorflow.keras import backend as k
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Multiply, Add, concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


# KL divergence computation
class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def calc(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * k.sum(1 + log_var - k.square(mu) - k.exp(log_var), axis=-1)
        self.add_loss(k.mean(kl_batch), inputs=inputs)
        return inputs


class VAE:
    def __init__(self, original_dim: int, latent_dim: int, batch_size: int, intermediate_dim1: int,
                 intermediate_dim2: int, intermediate_dim3: int, intermediate_dim4: int, lr: float, epochs: int,
                 activation: Layer, out_activation: str, validation_split: float, optimizer: OptimizerV2,
                 kernel_initializer: str, bias_initializer: str, checkpoint_dir: str,
                 early_stop: bool):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.intermediate_dim1 = intermediate_dim1
        self.intermediate_dim2 = intermediate_dim2
        self.intermediate_dim3 = intermediate_dim3
        self.intermediate_dim4 = intermediate_dim4
        self.lr = lr
        self.epochs = epochs
        self.activation = activation
        self.out_activation = out_activation
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.checkpoint_dir = checkpoint_dir
        self.early_stop = early_stop

        # Build the encoder
        x_in = Input((self.original_dim,))
        e_cond = Input(shape=(1,))
        angle_cond = Input(shape=(1,))
        geo_cond = Input(shape=(2,))
        merged_input = concatenate([x_in, e_cond, angle_cond, geo_cond], )
        h1 = Dense(self.intermediate_dim1, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(merged_input)
        h1 = BatchNormalization()(h1)
        h2 = Dense(self.intermediate_dim2, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h1)
        h2 = BatchNormalization()(h2)
        h3 = Dense(self.intermediate_dim3, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h2)
        h3 = BatchNormalization()(h3)
        h4 = Dense(self.intermediate_dim4, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h3)
        h = BatchNormalization()(h4)
        z_mu = Dense(self.latent_dim, )(h)
        z_log_var = Dense(self.latent_dim, )(h)
        # compute the KL divergence
        z_mu, z_log_var = KLDivergenceLayer().calc([z_mu, z_log_var])
        # Reparameterization trick
        z_sigma = Lambda(lambda t: k.exp(.5 * t))(z_log_var)
        eps = Input(tensor=k.random_normal(shape=(k.shape(x_in)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        z_cond = concatenate([z, e_cond, angle_cond, geo_cond], )
        # This defines the encoder which takes noise and input and outputs the latent variable z
        self.encoder = Model(inputs=[x_in, e_cond, angle_cond, geo_cond, eps], outputs=z_cond)
        # Build the decoder / Generator
        deco_l4 = Dense(self.intermediate_dim4, input_dim=(self.latent_dim + 4),
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l4_bn = BatchNormalization()
        deco_l3 = Dense(self.intermediate_dim3, input_dim=self.intermediate_dim4,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l3_bn = BatchNormalization()
        deco_l2 = Dense(self.intermediate_dim2, input_dim=self.intermediate_dim3,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l2_bn = BatchNormalization()
        deco_l1 = Dense(self.intermediate_dim1, input_dim=self.intermediate_dim2,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l1_bn = BatchNormalization()
        x_reco = Dense(self.original_dim, activation=self.out_activation)
        z_deco_input = Input(shape=(self.latent_dim + 4,))
        x_reco_deco = x_reco(
            (deco_l1_bn(deco_l1(deco_l2_bn(deco_l2(deco_l3_bn(deco_l3(deco_l4_bn(deco_l4(z_deco_input))))))))))
        # This defines the decoder which takes an input of size latent dimension + condition size dimension and outputs
        # the  reconstructed input version
        self.decoder = Model(inputs=[z_deco_input], outputs=[x_reco_deco])

        # This defines the reconstruction loss of the VAE model
        def _reconstruction_loss(g4_event, vae_event):
            return k.mean(self.original_dim * k.sum(metrics.binary_crossentropy(g4_event, vae_event)))

        # This defines the VAE model (encoder and decoder)
        self.vae = Model(inputs=[x_in, e_cond, angle_cond, geo_cond, eps],
                         outputs=[self.decoder(self.encoder([x_in, e_cond, angle_cond, geo_cond, eps]))])
        self.vae.compile(optimizer=self.optimizer, loss=[_reconstruction_loss])

    # Training function
    def train(self, train_set, e_cond, angle_cond, geo_cond, verbose=True):
        # If the early stopping flag is on then stop the training when a monitored metric (validation) has stopped
        # improving after (patience) number of epochs
        if self.early_stop:
            c_p = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=5, verbose=1)
        # If the early stopping flag is off then run the training for the number of epochs and save the model
        # every (period) epochs
        else:
            c_p = ModelCheckpoint(f"{self.checkpoint_dir}VAE-{{epoch:02d}}.h5", monitor="val_loss",
                                  verbose=0, save_best_only=False, save_weights_only=False,
                                  mode="min",
                                  save_freq=100)  # the model will be saved every 100 epochs
        noise = numpy.random.normal(0, 1, size=(train_set.shape[0], self.latent_dim))
        history = self.vae.fit([train_set, e_cond, angle_cond, geo_cond, noise], [train_set],
                               shuffle=True,
                               epochs=self.epochs,
                               verbose=verbose,
                               validation_split=self.validation_split,
                               batch_size=self.batch_size,
                               callbacks=[c_p]
                               )
        return history

    # # Encode function uses only the encoder to generate the latent representation of an input
    # def encode(self, dataSet):
    #     return self.encoder.predict(dataSet, batch_size=self.batch_size)
    # # Generate function uses only the decoder to generate new showers using the z_sample which is a vector of
    # # ND Gaussians
    # def generate(self, z_sample):
    #     return self.decoder.predict([z_sample])
    # # Predict function
    # def predict(self, dataSet):
    #     return self.vae.predict(dataSet, batch_size=self.batch_size)
    # # Evaluate function
    # def evaluate(self, dataSet):
    #     return self.vae.evaluate(dataSet, batch_size=self.batch_size)
