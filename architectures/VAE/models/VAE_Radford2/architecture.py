import os

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten, Reshape, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from . import config as C

from core.SaveLoad import LoadWeights

class VAE:
  """ Defines the VAE Architecture"""
  def __init__(self):
    channels = 3 if C.COLOR_MODE == 'rgb' else 1

    self.inputs = Input(shape=(C.IMG_SIZE, C.IMG_SIZE, channels))
    self.encoder = self._buildEncoder(self.inputs)
    self.decoder = self._buildDecoder()
    self.combined = self._buildCombined()

    self.optimizer = Adam(lr=C.LEARNING_RATE, beta_1=0.5, amsgrad=True)

    self.combined.compile(optimizer = self.optimizer, loss = self._loss_function)


  def _buildEncoder(self, input_tensor):
    """ Builds the Encoder Model """
    # returns: ( mu(input_tensor), log_sigma(input_tensor) )

    x = input_tensor
    # (64, 64, 3)

    x = Conv2D(64, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (32, 32, 64)

    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (16, 16, 128)

    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (8, 8, 256)

    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (4, 4, 512)

    x = Flatten()(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)

    mu = Dense(C.Z_LAYER_SIZE, activation='linear')(x)
    log_sigma = Dense(C.Z_LAYER_SIZE, activation='linear')(x)

    encoder = Model(input_tensor, [mu, log_sigma], name='Encoder')

    return encoder

  def _buildDecoder(self):
    """ Builds the Decoder Model """
    z = Input(shape=(C.Z_LAYER_SIZE,))
    x = z
    # (z)

    x = Dense(1000, activation='tanh')(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)
    # (1000)

    x = Reshape((1, 1, 1000))(x)
    # (1,1,1000)

    x = Conv2DTranspose(512, 4, strides=1, activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (4, 4, 512)

    x = Conv2DTranspose(512, 5, strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (8, 8, 512)

    x = Conv2DTranspose(512, 5, strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (16, 16, 512)

    x = Conv2DTranspose(512, 5, strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (32, 32, 512)

    x = Conv2DTranspose(3, 5, strides=2, activation='sigmoid', padding='same')(x)
    # (64, 64, 3)

    decoder = Model(z, x, name='Decoder')

    return decoder

  def _sample_z(self, args):
    """ Samples z from the learnt distribution """
    mu, log_sigma = args
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    eps = K.random_normal(shape=(K.shape(mu)[0], C.Z_LAYER_SIZE), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mu + K.exp(log_sigma) * eps

  def _buildCombined(self):
    """ Returns the VAE Model """
    # VAE model, for reconstruction and training

    encoded_distribution = self.encoder(self.inputs)
    z = Lambda(self._sample_z)(encoded_distribution)

    # decoder outputs = f(z)
    outputs = self.decoder(z)

    # vae outputs = f(inputs)
    vae = Model(self.inputs, outputs, name='VAE')

    return vae

  def _loss_function(self, inputs_flat, outputs_flat):
    """ The VAE Loss Function used during training """
    mu, log_sigma = self.encoder(self.inputs)
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(outputs_flat - inputs_flat))

    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.square(K.exp(log_sigma)), axis=-1)

    # return the average loss over all images in batch
    total_loss = K.mean((1 - C.KL_FACTOR ) * reconstruction_loss + C.KL_FACTOR * kl_loss)
    return total_loss

  def printSummary(self):
    print('\n VAE Summary:')
    self.combined.summary()
    
    print('\n Encoder Summary:')
    self.encoder.summary()
    
    print('\n Decoder Summary:')
    self.decoder.summary()

  def save(self, target_directory):
    self.combined.save(os.path.join(target_directory, 'model.h5'))
    self.combined.save_weights(os.path.join(target_directory,'weights.h5'))
    
    self.encoder.save(os.path.join(target_directory, 'encoder_model.h5'))
    self.encoder.save_weights(os.path.join(target_directory,'encoder_weights.h5'))
    
    self.decoder.save(os.path.join(target_directory,'decoder_model.h5'))
    self.decoder.save_weights(os.path.join(target_directory,'decoder_weights.h5'))

    print('Model & Weights Saved to: {}'.format(target_directory))

  def load(self, from_directory):
    LoadWeights(self.combined, os.path.join(from_directory, 'weights.hdf5'))