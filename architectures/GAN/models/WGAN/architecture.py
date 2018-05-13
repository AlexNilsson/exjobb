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

class WGAN_GP:
  """ Defines the Improved WGAN (GP) Architecture"""
  # https://arxiv.org/abs/1704.00028

  def __init__(self):
    self.channels = 3 if C.COLOR_MODE == 'rgb' else 1
    self.img_shape = (C.IMG_SIZE, C.IMG_SIZE, self.channels)
    self.dropout = C.DROPUT_AMOUNT

    self.input = Input(shape=(C.Z_LAYER_SIZE,))
    self.generator = self._buildGenerator(self.input)
    self.critic = self._buildCritic()
    #self.combined = self._buildCombined()

    # Optimizer
    optimizer = Adam(2e-04, beta_1=0.5, beta_2=0.9, amsgrad=True)

    self.generator.trainable = False

    # Image input (real sample)
    real_img = Input(shape=self.img_shape)

    # Image output (fake sample)
    fake_img = self.generator

    # Critic determines validity of real and fake images
    fake = self.critic(fake_img)
    valid = self.critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    # Determine validity of weighted sample
    validity_interpolated = self.critic(interpolated_img)

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(self._gradient_penalty_loss, averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    self.critic_model = Model(inputs=[real_img, self.input], outputs=[valid, fake, validity_interpolated])
    self.critic_model.compile(loss=[self._wasserstein_loss, self._wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])


    # compile Generator model
    self.generator.compile(loss = self._wasserstein_loss, optimizer = optimizer_g)

    # compile Critic model
    self.critic.compile(loss = 'binary_crossentropy', optimizer = optimizer_d)

    # compile Combined model, with frozen discriminator
    self.combined.get_layer(name='Critic').trainable = False
    self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer_g)

  def _buildGenerator(self, input_noise):
    """ Builds the Generator Model """
    # G(noise): noise -> img

    x = input_noise

    x = Reshape((1, 1, 100))(x)
    # (1, 1, 100)

    x = Conv2DTranspose(512, 4, strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (4, 4, 512)

    #? Block 1
    x = Conv2DTranspose(512, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (8, 8, 512)

    #? Block 2
    x = Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (16, 16, 256)

    #? Block 3
    x = Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (32, 32, 128)

    #? Block 4
    x = Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    # (64, 64, 64)

    x = Conv2DTranspose(3, 5, strides=2, activation='tanh', padding='same')(x)
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

    x = Conv2D(128, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (32, 32, 128)

    x = Conv2D(256, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (16, 16, 256)

    x = Conv2D(512, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (8, 8, 512)

    x = Conv2D(512, 5, strides=2, kernel_initializer='he_normal', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (4, 4, 512)

    x = Flatten()(x)
    x = Dropout(self.dropout)(x)
    # (1, 1, 8192)
    
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = LeakyReLU()(x)
    x = Dropout(self.dropout)(x)
    # (1, 1, 1024)

    x = Dense(1, kernel_initializer='he_normal')(x)
    # (1, 1, 1)

    score = x

    critic = Model(img, score, name='Critic')

    return critic

  def _buildCombined(self):
    """ Builds the Combined Model """
    # G(noise): noise -> p_real

    generated_img = self.generator(self.input)
    img_validity = self.critic(generated_img)

    combined = Model(self.input, img_validity, name='Combined')

    return combined

  def _wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

  def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)

    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)

    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)

  def printSummary(self):
    print('\n GAN Summary:')
    self.combined.summary()
    
    print('\n Generator Summary:')
    self.generator.summary()
    
    print('\n Critic Summary:')
    self.critic.summary()

  def save(self, target_directory):
    self.combined.save(os.path.join(target_directory, 'model.h5'))
    self.combined.save_weights(os.path.join(target_directory,'weights.h5'))

    self.generator.save(os.path.join(target_directory, 'generator_model.h5'))
    self.generator.save_weights(os.path.join(target_directory,'generator_weights.h5'))

    self.critic.save(os.path.join(target_directory,'critic_model.h5'))
    self.critic.save_weights(os.path.join(target_directory,'critic_weights.h5'))

    print('Model & Weights Saved to: {}'.format(target_directory))

  def load(self, from_directory):
    LoadWeights(self.combined, os.path.join(from_directory, 'weights.hdf5'))


def train(self, epochs, batch_size, sample_interval=50):
  # Load the dataset
  (X_train, _), (_, _) = mnist.load_data()

  # Rescale -1 to 1
  X_train = (X_train.astype(np.float32) - 127.5) / 127.5
  X_train = np.expand_dims(X_train, axis=3)

  # Adversarial ground truths
  valid = -np.ones((batch_size, 1))
  fake =  np.ones((batch_size, 1))
  dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
  for epoch in range(epochs):

      for _ in range(self.n_critic):

          # ---------------------
          #  Train Discriminator
          # ---------------------

          # Select a random batch of images
          idx = np.random.randint(0, X_train.shape[0], batch_size)
          imgs = X_train[idx]
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





  def train(self, PATHS,  N_DATA, data_generator, batch_callbacks=[], epoch_callbacks=[]):

    batches = math.floor(N_DATA/C.BATCH_SIZE)

    # Arbitrary starting loss
    g_loss = d_loss = DISC_TRAIN_THRESH

    # running means
    d_loss_rm = np.ones(40)
    g_loss_rm = np.ones(40)

    for epoch in range(1, C.EPOCHS + 1):

      for callback in epoch_callbacks:
        callback.on_epoch_begin(epoch, None)

      for batch_idx in range(data_generator.__len__()):
        batch = data_generator.__getitem__(batch_idx)

        """ Train Discriminator """
        # Select a random half batch of images
        real_data = batch[np.random.randint(0, int(C.BATCH_SIZE), int(C.BATCH_SIZE/2))]
        real_data = (real_data*2) - 1 #[0,1] -> [-1,1]

        # Sample noise and generate a half batch of new images
        #flat_img_length = batch.shape[1]
        noise = np.random.normal(-1, 1, (int(C.BATCH_SIZE/2), int(C.Z_LAYER_SIZE)))
        fake_data = self.generator.predict(noise)

        # Train discriminator half as much as generator
        if train_d:
          # Train the discriminator (real classified as ones and generated as zeros), update loss accordingly
          d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((int(C.BATCH_SIZE/2), 1)))
          d_loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros((int(C.BATCH_SIZE/2), 1)))
        else:
          # Test the discriminator (real classified as ones and generated as zeros), update loss accordingly
          d_loss_real = self.discriminator.test_on_batch(real_data, np.ones((int(C.BATCH_SIZE/2), 1)))
          d_loss_fake = self.discriminator.test_on_batch(fake_data, np.zeros((int(C.BATCH_SIZE/2), 1)))

        d_loss = np.mean([d_loss_real, d_loss_fake])
        #d_loss_epoch.append(d_loss)

        """ Train Generator """
        # Sample generator input
        noise = np.random.normal(-1, 1, (int(C.BATCH_SIZE), int(C.Z_LAYER_SIZE)))

        if train_g:
          # Train the generator to fool the discriminator, e.g. classify these images as real (1)
          # The discriminator model is frozen in this stage but its gradient is still used to guide the generator
          g_loss = self.combined.train_on_batch(noise, np.ones((C.BATCH_SIZE, 1)))
        else:
          g_loss = self.combined.test_on_batch(noise, np.ones((C.BATCH_SIZE, 1)))

        #g_loss_epoch.append(g_loss)

        d_loss_rm = np.append(d_loss_rm, d_loss+0.5)
        d_loss_rm = d_loss_rm[-40:]

        g_loss_rm = np.append(g_loss_rm, g_loss+0.5)
        g_loss_rm = g_loss_rm[-40:]

        # Plot the progress
        print("Epoch:{} Batch:{}/{} [D loss: {}] [G loss: {}]".format(epoch, batch_idx+1, batches, d_loss, g_loss))
        
        for callback in batch_callbacks:
          callback.on_batch_end(epoch*1000 + batch_idx+1, {'d_loss': np.mean(d_loss_rm), 'g_loss': np.mean(g_loss_rm)})

      # Epoch on_end Callbacks
      for callback in epoch_callbacks:
          callback.on_epoch_end(epoch, {'d_loss': np.mean(d_loss_rm), 'g_loss': np.mean(g_loss_rm)})

      if epoch % C.SAVE_WEIGHTS_FREQ == 0:
        self.combined.save_weights(os.path.join(PATHS.PATH_TO_SAVED_WEIGHTS,'combined_weight.hdf5'))