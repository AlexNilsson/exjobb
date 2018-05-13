import os
import math
import numpy as np

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, Conv2DTranspose, ZeroPadding2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from core.SaveLoad import LoadWeights

from . import config as C

class GAN:
  """ Defines the GAN Architecture"""
  def __init__(self):
    self.channels = 3 if C.COLOR_MODE == 'rgb' else 1
    self.img_shape = (C.IMG_SIZE, C.IMG_SIZE, self.channels)
    self.dropout = C.DROPUT_AMOUNT

    self.input = Input(shape=(C.Z_LAYER_SIZE,))
    self.generator = self._buildGenerator(self.input)
    self.discriminator = self._buildDiscriminator()
    self.combined = self._buildCombined()

    # Optimizer
    optimizer_g  = Adam(0.0002, 0.5, amsgrad=True)
    optimizer_d = Adam(0.0004, 0.5, amsgrad=True)

    # compile Generator model
    self.generator.compile(loss = 'binary_crossentropy', optimizer = optimizer_g)

    # compile Discriminator model
    self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer_d)

    # compile Combined model, with frozen discriminator
    self.combined.get_layer(name='Discriminator').trainable = False
    self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer_g)

  def _buildGenerator(self, input_noise):
    """ Builds the Generator Model """
    # G(noise): noise -> img

    x = input_noise

    x = Reshape((1, 1, 100))(x)
    # (1, 1, 100)

    x = Conv2DTranspose(512, 4, strides=2, activation='tanh')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (4, 4, 512)

    x = Conv2DTranspose(256, 5, strides=2, activation='tanh', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (8, 8, 256)

    x = Conv2DTranspose(128, 5, strides=2, activation='tanh', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (16, 16, 128)

    x = Conv2DTranspose(64, 5, strides=2, activation='tanh', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (32, 32, 64)

    x = Conv2DTranspose(64, 3, activation='tanh', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (64, 64, 64)

    x = Conv2DTranspose(3, 5, strides=2, activation='tanh', padding='same')(x)
    # (64, 64, 3)

    img = x

    generator = Model(input_noise, img, name='Generator')

    return generator

  def _buildDiscriminator(self):
    """ Builds the Discriminator Model """
    # D(img): img -> p_real

    img = Input(shape=self.img_shape)
    x = img
    # (64, 64, 3)

    x = Conv2D(64, 5, strides=2, activation='tanh', padding='same')(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (32, 32, 64)

    x = Conv2D(128, 5, strides=2, activation='tanh', padding='same')(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (16, 16, 128)

    x = Conv2D(256, 5, strides=2, activation='tanh', padding='same')(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (8, 8, 256)

    x = Conv2D(512, 5, strides=2, activation='tanh', padding='same')(x)
    #x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    # (4, 4, 512)

    x = Flatten()(x)
    x = Dropout(self.dropout)(x)
    x = Dense(1, activation='sigmoid')(x)

    validity = x

    descriminator = Model(img, validity, name='Discriminator')

    return descriminator

  def _buildCombined(self):
    """ Builds the Combined Model """
    # G(noise): noise -> p_real

    generated_img = self.generator(self.input)
    img_validity = self.discriminator(generated_img)

    combined = Model(self.input, img_validity, name='Combined')

    return combined

  def printSummary(self):
    print('\n GAN Summary:')
    self.combined.summary()
    
    print('\n Generator Summary:')
    self.generator.summary()
    
    print('\n Discriminator Summary:')
    self.discriminator.summary()

  def save(self, target_directory):
    self.combined.save(os.path.join(target_directory, 'model.h5'))
    self.combined.save_weights(os.path.join(target_directory,'weights.h5'))

    self.generator.save(os.path.join(target_directory, 'generator_model.h5'))
    self.generator.save_weights(os.path.join(target_directory,'generator_weights.h5'))

    self.discriminator.save(os.path.join(target_directory,'discriminator_model.h5'))
    self.discriminator.save_weights(os.path.join(target_directory,'discriminator_weights.h5'))

    print('Model & Weights Saved to: {}'.format(target_directory))

  def load(self, from_directory):
    LoadWeights(self.combined, os.path.join(from_directory, 'weights.hdf5'))

  def train(self, PATHS,  N_DATA, data_generator, batch_callbacks=[], epoch_callbacks=[]):

    DISC_TRAIN_THRESH = -math.log(0.5) # ~cross entropy loss at 50% correct classification

    batches = math.floor(N_DATA/C.BATCH_SIZE)

    # Arbitrary starting loss
    g_loss = d_loss = DISC_TRAIN_THRESH

    # running means
    d_loss_rm = np.ones(40)
    g_loss_rm = np.ones(40)

    d_turn = True
    train_d, train_g = True, True

    for epoch in range(1, C.EPOCHS + 1):

      for callback in epoch_callbacks:
        callback.on_epoch_begin(epoch, None)

      for batch_idx in range(data_generator.__len__()):
        batch = data_generator.__getitem__(batch_idx)

        if d_turn:
          train_d = True
          train_g = False
          if batch_idx % 5 == 0:
            train_d = False
            train_g = True
            d_turn = False
        else:
          train_d = False
          train_g = True
          if batch_idx % 1 == 0:
            train_d = True
            train_g = False
            d_turn = True

        if np.mean(d_loss_rm[-20:]) > np.mean(g_loss_rm[-20:]):
          train_d = True
          train_g = False

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