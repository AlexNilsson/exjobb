import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import config as C

# Custom generator to create batch pairs so we can compare y to x, without labels
# Returns (x_batch, y_batch)
def data_pair_generator(generator):
  for batch in generator:
    batch = batch / 255
    batch_ref = batch
    if C.NOISE_FACTOR > 0:
      batch_ref = batch_ref + C.NOISE_FACTOR * np.random.normal(size=batch_ref.shape)
    yield (batch, batch_ref)

def getDataPairGenerator(path_to_dataset):
  data_augmentor = ImageDataGenerator(
    #zoom_range = C.ZOOM_RANGE,
    #channel_shift_range = C.CHANNEL_SHIFT_RANGE,
    #horizontal_flip = C.HORIZONTAL_FLIP,
    #data_format = 'channels_last'
  )

  data_generator = data_augmentor.flow_from_directory(
    path_to_dataset,
    target_size = (C.IMG_SIZE, C.IMG_SIZE),
    batch_size = C.BATCH_SIZE,
    class_mode = None,
    color_mode = C.COLOR_MODE
  )

  return data_pair_generator(data_generator)
