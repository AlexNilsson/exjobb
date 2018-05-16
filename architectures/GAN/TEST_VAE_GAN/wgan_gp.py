
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division
import os

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt
import cv2 as cv

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..','..'))

import numpy as np

from core.Generators import Generators
from core.Visualizer import Visualizer
from core.SaveLoad import LoadWeights
import config as C

class RandomWeightedAverage(_Merge):
  """Provides a (random) weighted average between real and generated image samples"""
  def _merge_function(self, inputs):
    alpha = K.random_uniform((C.BATCH_SIZE, 1, 1, 1))
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
  def __init__(self):
    self.channels = 3 if C.COLOR_MODE == 'rgb' else 1
    self.img_shape = (C.IMG_SIZE, C.IMG_SIZE, self.channels)
    self.latent_dim = C.Z_LAYER_SIZE

    self.PATH_TO_THIS_DIR = os.path.dirname(__file__)

    # Load pretrained vae_decoder
    self.vae_decoder = load_model(os.path.join(self.PATH_TO_THIS_DIR, '..', C.VAE_DECODER,'decoder_model.h5'))
    #self.vae_decoder.load(os.path.join(self.PATH_TO_THIS_DIR, '..', C.VAE_DECODER,'decoder_weights.h5'))
    LoadWeights(self.vae_decoder, os.path.join(self.PATH_TO_THIS_DIR, '..', C.VAE_DECODER,'decoder_weights.h5'))
    self.vae_decoder.trainable = False

    # Following parameter and optimizer set as recommended in paper
    self.n_critic = C.N_TRAIN_CRITIC
    optimizer = RMSprop(lr=0.00001)

    # Build the generator and critic
    self.generator = self.build_generator()
    self.visualizer = Visualizer(C) 
    self.critic = self.build_critic()
    
    

    #-------------------------------
    # Construct Computational Graph
    #       for the Critic
    #-------------------------------

    # Freeze generator's layers while training critic
    self.generator.trainable = False

    # Image input (real sample)
    real_img = Input(shape=self.img_shape)

    # Noise input
    z_disc = Input(shape=(100,))
    # Generate image based of noise (fake sample)
    fake_img = self.generator(z_disc)

    # Discriminator determines validity of the real and fake images
    fake = self.critic(fake_img)
    valid = self.critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    # Determine validity of weighted sample
    validity_interpolated = self.critic(interpolated_img)

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(self.gradient_penalty_loss,
                      averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    self.critic_model = Model(inputs=[real_img, z_disc],
                        outputs=[valid, fake, validity_interpolated])
    self.critic_model.compile(loss=[self.wasserstein_loss,
                                          self.wasserstein_loss,
                                          partial_gp_loss],
                                    optimizer=optimizer,
                                    loss_weights=[1, 1, C.GRADIENT_PENALTY_WEIGHT])
    #-------------------------------
    # Construct Computational Graph
    #         for Generator
    #-------------------------------

    # For the generator we freeze the critic's layers
    self.critic.trainable = False
    self.generator.trainable = True

    # Sampled noise for input to generator
    z_gen = Input(shape=(100,))
    # Generate images based of noise
    img = self.generator(z_gen)
    # Discriminator determines validity
    valid = self.critic(img)
    # Defines generator model
    self.generator_model = Model(z_gen, valid)
    self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


  def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


  def wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

  def build_generator(self):

    noise = Input(shape=(self.latent_dim,))
    # (100)

    x = self.vae_decoder(noise)
    # (64, 64, 3)

    x = UpSampling2D()(x)
    x = Conv2D(256, kernel_size=4, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (128, 128, 256)

    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=4, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (256, 256, 128)
    
    x = Conv2D(self.channels, kernel_size=4, padding="same")(x)
    img = Activation('tanh')(x)
    # (512, 512, 3)
    
    return Model(noise, img)

  def build_critic(self):

    model = Sequential()

    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Flatten())
    model.add(Dense(1))

    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)

  def train(self, epochs, batch_size, sample_interval=50):

    # Load the dataset
    PATH_TO_DATA_DIR = os.path.join(self.PATH_TO_THIS_DIR, '../../../data')
    PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)
    data_generator = Generators(C).getDataGenerator(PATH_TO_DATASET)

    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake =  np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
    for epoch in range(epochs):
      for _ in range(self.n_critic):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Get data
        imgs = next(data_generator)
        if imgs.shape[0] < C.BATCH_SIZE:
          # sample another batch if the final batch is too small
          imgs = next(data_generator)
        # Rescale -1 to 1
        imgs = imgs / 255 * 2 - 1

        # Sample generator input
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        # Train the critic
        d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                        [valid, fake, dummy])

      # ---------------------
      #  Train Generator
      # ---------------------

      # Sample generator input
      noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
      # Train the generator
      g_loss = self.generator_model.train_on_batch(noise, valid)

      # Plot the progress
      print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

      # If at save interval => save generated image samples
      if epoch % sample_interval == 0:
          self.sample_images(epoch)

  def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    #gen_imgs = self.generator.predict(noise)

    figure = self.visualizer.getLatentSpaceGrid(self.generator, noise)
    file_path = os.path.join(self.PATH_TO_THIS_DIR, 'images', 'mnist_{}.jpg'.format(epoch))
    cv.imwrite(file_path, figure)


if __name__ == '__main__':
  wgan = WGANGP()
  wgan.train(epochs=30000, batch_size=C.BATCH_SIZE, sample_interval=10)