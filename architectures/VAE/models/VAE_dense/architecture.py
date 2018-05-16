from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from . import config as C

class VAE:
  """ Defines the VAE Architecture"""
  def __init__(self):
    channels = 3 if C.COLOR_MODE == 'rgb' else 1

    self.inputs = Input(shape=(C.IMG_SIZE*C.IMG_SIZE,))
    self.encoder = self.buildEncoder(self.inputs)
    self.decoder = self.buildDecoder()
    self.combined = self.buildModel()

    self.optimizer = Adam(lr=C.LEARNING_RATE, amsgrad=True)

    self.combined.compile(optimizer = self.optimizer, loss = self.loss_function)

  def buildEncoder(self, input_tensor):
    """ Builds the Encoder Model """
    # returns: ( mu(input_tensor), log_sigma(input_tensor) )
    x = input_tensor


    x = Dense(200, activation='relu')(x)
    x = Dense(100, activation='relu')(x)

    mu = Dense(C.Z_LAYER_SIZE, activation='linear')(x)
    log_sigma = Dense(C.Z_LAYER_SIZE, activation='linear')(x)

    encoder = Model(input_tensor, [mu, log_sigma], name='Encoder')

    return encoder

  def buildDecoder(self):
    """ Builds the Decoder Model """
    z = Input(shape=(C.Z_LAYER_SIZE,))
    x = z



    x = Dense(100, activation='relu')(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(C.DROPUT_AMOUNT)(x)
    x = Dense(784, activation='relu')(x)
    


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