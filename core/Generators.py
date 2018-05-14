import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class Generators:
  def __init__(self, config):
    self.config = config

  def getDataGenerator(self, path_to_dataset):
    C = self.config
    data_augmentor = ImageDataGenerator(
      zoom_range = C.ZOOM_RANGE,
      channel_shift_range = C.CHANNEL_SHIFT_RANGE,
      horizontal_flip = C.HORIZONTAL_FLIP,
      #data_format = 'channels_last'
    )

    data_generator = data_augmentor.flow_from_directory(
      path_to_dataset,
      target_size = (C.IMG_SIZE, C.IMG_SIZE),
      batch_size = C.BATCH_SIZE,
      class_mode = None,
      color_mode = C.COLOR_MODE
    )

    return data_generator

  # Custom generator to create batch pairs so we can compare y to x, without labels
  # Returns (x_batch, y_batch)
  def data_pair_generator(self, generator):
    C = self.config
    for batch in generator:
      batch = batch / 255
      batch_ref = batch
      if C.NOISE_FACTOR > 0:
        batch_ref = batch_ref + C.NOISE_FACTOR * np.random.normal(size=batch_ref.shape)
      yield (batch, batch_ref)

  def getDataPairGenerator(self, path_to_dataset):

    data_generator = self.getDataGenerator(path_to_dataset)

    return self.data_pair_generator(data_generator)
