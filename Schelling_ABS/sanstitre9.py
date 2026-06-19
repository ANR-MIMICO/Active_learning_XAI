import numpy as np
from tensorflow.keras import layers, backend as K, Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.datasets import mnist
from sklearn.base import BaseEstimator, TransformerMixin

class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAEModel(Model):
    def __init__(self, encoder, decoder, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # compute losses
        recon_loss = binary_crossentropy(inputs, reconstructed) * self.input_dim
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        total_loss = K.mean(recon_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed

class VariationalAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim=5, latent_dim=2, intermediate_dim=512, learning_rate=0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.learning_rate = learning_rate
        self._build_model()

    def _build_model(self):
        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = layers.Dense(self.intermediate_dim, activation='relu')(encoder_inputs)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling(name='z')([z_mean, z_log_var])
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        x_dec = layers.Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        decoder_outputs = layers.Dense(self.input_dim, activation='sigmoid')(x_dec)
        decoder = Model(latent_inputs, decoder_outputs, name='decoder')

        # VAE as subclassed model
        self.vae = VAEModel(encoder, decoder, self.input_dim, name='vae')
        self.encoder = encoder
        self.decoder = decoder

        self.vae.compile(optimizer='adam')

    def fit(self, X, y=None, epochs=50, batch_size=128, validation_split=0.1, **kwargs):
#        X = X.reshape(-1, self.input_dim).astype('float32') / 255.
        X = X.reshape(-1, 5)
        self.vae.fit(X, X,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=validation_split,
                     **kwargs)
        return self

    def transform(self, X):
        X = X.reshape(-1, self.input_dim).astype('float32') / 255.
        z_mean, _, _ = self.encoder.predict(X)
        return z_mean

    def inverse_transform(self, Z):
        return self.decoder.predict(Z)

if __name__ == '__main__':
    (x_train, _), _ = mnist.load_data()
    x_train = np.random.rand(50,5)
    x_train[:,-1] = np.sum(x_train[:,:-1],axis=1)
    
    vae = VariationalAutoencoder(latent_dim=2)
    vae.fit(x_train, epochs=50)
    z = vae.transform(x_train)
    x_decoded = vae.inverse_transform(z)

    z = vae.transform(x_train)
    x_decoded = vae.inverse_transform(z)
    




