from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LeakyReLU
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

    x = Dense(4 * 4 * 1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 1024))(x)
    x = BatchNormalization(momentum = 0.8)(x)

    x = UpSampling2D()(x)
    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)

    x = UpSampling2D()(x)
    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)

    x = UpSampling2D()(x)
    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)

    x = Conv2D(self.channels, 5, strides=2, activation='sigmoid', padding='same')(x)

    img = x

    generator = Model(input_noise, img, name='Generator')

    return generator

  def buildDiscriminator(self):
    """ Builds the Discriminator Model """
    # D(img): img -> p_real

    img = Input(shape=self.img_shape)
    x = img

    x = Conv2D(512, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(self.dropout)(x)

    x = Conv2D(256, 5, strides=2, padding='same')(x)
    x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(self.dropout)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(128, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(self.dropout)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(64, 5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(self.dropout)(x)

    x = Flatten()(x)
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
