from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, Conv2DTranspose, ZeroPadding2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU
from keras.models import Model

from . import config as C

class GAN:
  """ Defines the GAN Architecture"""
  def __init__(self):
    self.channels = 3 if C.COLOR_MODE == 'rgb' else 1
    self.img_shape = (C.IMG_SIZE, C.IMG_SIZE, self.channels)
    self.dropout = C.DROPUT_AMOUNT

    self.input = Input(shape=(C.Z_LAYER_SIZE,))
    self.generator = self.buildGenerator(self.input)
    self.discriminator = self.buildDiscriminator()
    self.combined = self.buildCombined()

  def buildGenerator(self, input_noise):
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

  def buildDiscriminator(self):
    """ Builds the Discriminator Model """
    # D(img): img -> p_real

    img = Input(shape=self.img_shape)
    x = img
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
    x = Dropout(self.dropout)(x)
    x = Dense(1, activation='sigmoid')(x)

    validity = x

    descriminator = Model(img, validity, name='Discriminator')

    return descriminator

  def buildCombined(self):
    """ Builds the Combined Model """
    # G(noise): noise -> p_real

    generated_img = self.generator(self.input)
    img_validity = self.discriminator(generated_img)

    combined = Model(self.input, img_validity, name='Combined')

    return combined
