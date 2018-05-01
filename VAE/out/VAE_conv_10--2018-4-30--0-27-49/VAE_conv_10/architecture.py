from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten, Reshape, LeakyReLU
from keras.models import Model
import keras.backend as K

from . import config as C

class VAE:
  """ Defines the VAE Architecture"""
  def __init__(self):
    channels = 3 if C.COLOR_MODE == 'rgb' else 1

    self.inputs = Input(shape=(C.IMG_SIZE, C.IMG_SIZE, channels))
    self.encoder = self.buildEncoder(self.inputs)
    self.decoder = self.buildDecoder()
    self.model = self.buildModel()

  def buildEncoder(self, input_tensor):
    """ Builds the Encoder Model """
    # returns: ( mu(input_tensor), log_sigma(input_tensor) )
    x = input_tensor

    x = Conv2D(64, 5, strides=2, activation="relu", padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(128, 5, strides=2, activation="relu", padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(256, 5, strides=2, activation="relu", padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(512, 5, strides=2, activation="relu", padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    convoluted = x

    # Variables used for decoders layers sizes
    self.convShape = convoluted.shape

    x = Flatten()(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)

    mu = Dense(C.Z_LAYER_SIZE, activation='linear')(x)
    log_sigma = Dense(C.Z_LAYER_SIZE, activation='linear')(x)

    encoder = Model(input_tensor, [mu, log_sigma], name='Encoder')

    return encoder

  def buildDecoder(self):
    """ Builds the Decoder Model """
    z = Input(shape=(C.Z_LAYER_SIZE,))
    x = z

    c1 = int(self.convShape[1])
    c2 = int(self.convShape[2])
    c3 = int(self.convShape[3])

    x = Dense(c1 * c2 * c3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)
    x = Reshape((c1, c2, c3))(x)
    
    x = Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization(momentum=0.7)(x)

    x = Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization(momentum=0.7)(x)

    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization(momentum=0.7)(x)

    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization(momentum=0.7)(x)

    x = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    decoder = Model(z, x, name='Decoder')

    return decoder

  def sample_z(self, args):
    """ Samples z from the learnt distribution """
    mu, log_sigma = args
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    eps = K.random_normal(shape=(K.shape(mu)[0], C.Z_LAYER_SIZE), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mu + K.exp(log_sigma) * eps

  def buildModel(self):
    """ Returns the VAE Model """
    # VAE model, for reconstruction and training

    encoded_distribution = self.encoder(self.inputs)
    z = Lambda(self.sample_z)(encoded_distribution)

    # decoder outputs = f(z)
    outputs = self.decoder(z)

    # vae outputs = f(inputs)
    vae = Model(self.inputs, outputs, name='VAE')

    return vae

  def loss_function(self, inputs_flat, outputs_flat):
    """ The VAE Loss Function used during training """
    mu, log_sigma = self.encoder(self.inputs)
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(outputs_flat - inputs_flat))

    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.square(K.exp(log_sigma)), axis=-1)

    # return the average loss over all images in batch
    total_loss = K.mean((1 - C.KL_FACTOR ) * reconstruction_loss + C.KL_FACTOR * kl_loss)
    return total_loss
