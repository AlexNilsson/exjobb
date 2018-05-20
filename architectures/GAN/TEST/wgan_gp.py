
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt
import cv2 as cv

import os
import sys

import numpy as np

from core.Generators import Generators
from core.Visualizer import Visualizer
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

    # Following parameter and optimizer set as recommended in paper
    self.n_critic = C.N_TRAIN_CRITIC
    optimizer = RMSprop(lr=0.00001)

    # Build the generator and critic
    self.generator = self.build_generator()
    self.visualizer = Visualizer(C) 
    self.critic = self.build_critic()
    
    self.PATH_TO_THIS_DIR = os.path.dirname(__file__)

    #-------------------------------
    # Construct Computational Graph
    #       for the Critic
    #-------------------------------

    # Freeze generator's layers while training critic
    self.generator.trainable = False

    # Image input (real sample)
    real_img = Input(shape=self.img_shape)

    # Noise input
    z_disc = Input(shape=(C.Z_LAYER_SIZE,))
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
    z_gen = Input(shape=(C.Z_LAYER_SIZE,))
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

    model = Sequential()

    model.add(Dense(512 * 4 * 4, activation="relu", input_dim=self.latent_dim))
    model.add(Reshape((4, 4, 512)))

    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(self.latent_dim,))
    img = model(noise)

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

    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(C.DROPUT_AMOUNT))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
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

      if epoch % 100 == 0:
        self.generator_model.save_weights(os.path.join(self.PATH_TO_THIS_DIR,'generator_weights.h5'))
        self.critic_model.save_weights(os.path.join(self.PATH_TO_THIS_DIR,'critic_weights.h5'))

      # If at save interval => save generated image samples
      if epoch % sample_interval == 0:
          self.sample_images(epoch)

  def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    #gen_imgs = self.generator.predict(noise)

    figure = self.visualizer.getLatentSpaceGrid(self.generator, noise, img_size=1280)
    file_path = os.path.join(self.PATH_TO_THIS_DIR, 'images', 'mnist_{}.jpg'.format(epoch))
    cv.imwrite(file_path, figure)

""" 
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 1

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,0])
        axs[i,j].axis('off')
        cnt += 1
    fig.savefig(os.path.join(self.PATH_TO_THIS_DIR, 'images', 'mnist_{}.png'.format(epoch)))
    plt.close() """


if __name__ == '__main__':
  wgan = WGANGP()
  wgan.train(epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, sample_interval=C.PLOT_LATENT_SPACE_EVERY)