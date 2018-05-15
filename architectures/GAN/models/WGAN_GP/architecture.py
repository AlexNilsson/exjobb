import os
import math
import numpy as np

from functools import partial

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, Conv2DTranspose, ZeroPadding2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU, Activation
from keras.layers.merge import _Merge
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from core.SaveLoad import LoadWeights

from . import config as C

class RandomWeightedAverage(_Merge):
  def _merge_function(self, inputs):
    weights = K.random_uniform((C.BATCH_SIZE, 1, 1, 1))
    return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class GAN:
  """ Defines the Improved WGAN (GP) Architecture"""
  # Implemented more or less according to
  # https://arxiv.org/abs/1704.00028
  # Gulrajani et al.
  # With most recommended settings

  def __init__(self):
    self.channels = 3 if C.COLOR_MODE == 'rgb' else 1
    self.img_shape = (C.IMG_SIZE, C.IMG_SIZE, self.channels)

    self.dropout = C.DROPUT_AMOUNT
    self.n_critic = C.N_TRAIN_CRITIC # training rato between critic : generator
    self.gradient_penalty_weight = C.GRADIENT_PENALTY_WEIGHT

    self.optimizer = Adam(2e-04, beta_1=0.5, beta_2=0.9, amsgrad=True)

    self.input = Input(shape=(C.Z_LAYER_SIZE,))
    self.generator = self._buildGenerator(self.input)
    print('\n Generator Summary:')
    self.generator.summary()

    self.critic = self._buildCritic()
    print('\n Critic Summary:')
    self.critic.summary()

    """ Computational Graph for Training Generator """
    # Freeze critic part of network to only train the generator
    self.critic.trainable = False
    self.generator.trainable = True

    self.generator_model = self._compileCombinedModel()

    print('\n GAN Combined Generator Summary:')
    self.generator_model.summary()

    """ Computational Graph for Training Critic """
    # Freeze generator part of network to only train the critic
    self.critic.trainable = True
    self.generator.trainable = False

    self.critic_model = self._compileCombinedModelGP()
    print('\n GAN Combined Critic Summary:')
    self.critic_model.summary()


  def _buildGenerator(self, input_noise):
    """ Builds the Generator Model """
    # G(noise): noise -> img

    x = input_noise

    x = Reshape((1, 1, 100))(x)
    # (1, 1, 100)

    x = Conv2DTranspose(64, 4, strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (4, 4, 512)

    #? Block 1
    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (8, 8, 512)

    #? Block 2
    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (16, 16, 256)

    #? Block 3
    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (32, 32, 128)

    #? Block 4
    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (64, 64, 64)

    x = Conv2DTranspose(3, 5, activation='tanh', padding='same')(x)
    # (64, 64, 3)

    img = x

    generator = Model(input_noise, img, name='Generator')

    return generator

  def _buildCritic(self):
    """ Builds the Discriminator Model """
    # D(img): img -> p_real

    img = Input(shape=self.img_shape)
    x = img
    # (64, 64, 3)

    x = Conv2D(64, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (64, 64, 64)

    x = Conv2D(64, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (32, 32, 128)

    x = Conv2D(64, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (16, 16, 256)

    x = Conv2D(64, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (8, 8, 256)

    x = Conv2D(64, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (4, 4, 256)

    x = Flatten()(x)
    x = Dropout(self.dropout)(x)
    # (1, 1, 4096)

    x = Dense(64, kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (1, 1, 256)

    x = Dense(1, kernel_initializer='he_normal')(x)
    # (1, 1, 1)

    score = x

    critic = Model(img, score, name='Critic')

    return critic

  def _compileCombinedModel(self):
    """ Builds the Combined Model, primarily for training the Generator  """
    # f: noise -> score_real

    fake_samples = self.generator(self.input)
    fake_samples_score = self.critic(fake_samples)

    combined = Model(self.input, fake_samples_score, name='Combined')
    combined.compile(loss=self._wasserstein_loss, optimizer=self.optimizer)

    return combined

  def _compileCombinedModelGP(self):
    """ Builds the Combined Model, primarily for training the Critic  """
    # f: (real_samples, fake_samples) -> (score_real, score_fake, score_avg)

    # Real samples from dataset
    real_samples = Input(shape=self.img_shape)
    # Generated samples from noise
    fake_samples = self.generator(self.input)
    # Construct weighted averages between real and fake samples
    averaged_samples = RandomWeightedAverage()([real_samples, fake_samples])

    # Critic determines validity of real, fake and averaged images
    real_samples_score = self.critic(real_samples)
    fake_samples_score = self.critic(fake_samples)
    averaged_samples_score = self.critic(averaged_samples)

    # Use Python partial to provide loss function with additional 'averaged_samples' argument
    # as Keras loss functions can only have two arguments, y_true and y_pred.
    partial_gp_loss = partial(self._gradient_penalty_loss, averaged_samples=averaged_samples)
    partial_gp_loss.__name__ = 'gradient_penalty' # Function names are required by Keras #TODO: is this still the case?

    combined_gp = Model(inputs=[real_samples, self.input], outputs=[real_samples_score, fake_samples_score, averaged_samples_score])
    combined_gp.compile(loss=[self._wasserstein_loss, self._wasserstein_loss, partial_gp_loss], optimizer=self.optimizer, loss_weights=[1, 1, self.gradient_penalty_weight])

    return combined_gp

  def _wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

  def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)

    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)

    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)

  def printSummary(self):
    print('\n GAN Combined Generator Summary:')
    self.generator_model.summary()

    print('\n GAN Combined Critic Summary:')
    self.critic_model.summary()

    print('\n Generator Summary:')
    self.generator.summary()

    print('\n Critic Summary:')
    self.critic.summary()

  def save(self, target_directory):
    # generator_model containes the entire network, so use it to save/load everything
    self.generator_model.save(os.path.join(target_directory, 'model.h5'))
    self.generator_model.save_weights(os.path.join(target_directory,'weights.h5'))

    self.generator.save(os.path.join(target_directory, 'generator.h5'))
    self.generator.save_weights(os.path.join(target_directory,'generator_weights.h5'))

    self.critic.save(os.path.join(target_directory,'critic.h5'))
    self.critic.save_weights(os.path.join(target_directory,'critic_weights.h5'))

    print('Model & Weights Saved to: {}'.format(target_directory))

  def load(self, from_directory):
    LoadWeights(self.generator_model, os.path.join(from_directory, 'weights.hdf5'))

  def train(self, PATHS, data_generator, epoch_callbacks=[], batch_callbacks=[]):
    # Adversarial ground truths
    target_real_score = -np.ones((C.BATCH_SIZE, 1))
    target_fake_score =  np.ones((C.BATCH_SIZE, 1))
    dummy_score = np.zeros((C.BATCH_SIZE, 1)) # For gradient penalty

    # Initial loss
    c_loss = g_loss = math.inf

    for epoch in range(C.EPOCHS):

      for callback in epoch_callbacks:
        callback.on_epoch_begin(epoch, None)

      for batch in range(self.n_critic):
        """ Train Critic """
        # Real samples
        real_samples = next(data_generator)
        if real_samples.shape[0] < C.BATCH_SIZE:
          # sample another batch if the final batch is too small
          real_samples = next(data_generator)

        # rescale values from [0, 255] to domain [-1, 1]
        real_samples = 2 * real_samples / 255 - 1

        # Sample generator input
        noise = np.random.normal(0, 1, (C.BATCH_SIZE, C.Z_LAYER_SIZE))
        # Train the critic
        c_loss = self.critic_model.train_on_batch([real_samples, noise],[target_real_score, target_fake_score, dummy_score])

        # Print Progress
        print("Epoch:{} Batch:{}/{} [C loss: {}] [G loss: {}]".format(epoch, batch+1, self.n_critic, c_loss, g_loss))

      """ Train Generator """
      # Sample generator input
      noise = np.random.normal(0, 1, (C.BATCH_SIZE, C.Z_LAYER_SIZE))
      # Train the generator
      g_loss = self.generator_model.train_on_batch(noise, target_real_score)

      # Epoch on_end Callbacks
      for callback in epoch_callbacks:
        logs = {'d_loss': c_loss[0], 'g_loss': g_loss }
        callback.on_epoch_end(epoch, logs)

      # If at save interval => save generated image samples #TODO add as a callback?
      if epoch % C.SAVE_WEIGHTS_FREQ == 0:
        self.generator_model.save_weights(os.path.join(PATHS.PATH_TO_SAVED_WEIGHTS,'weights.hdf5'))
