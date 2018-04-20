import os
import numpy as np
from PIL import Image

def preProcessImages(in_directory, out_directory, convert_to_grayscale=False, resize_to=(40,40)):
  pot_files = os.listdir(in_directory)
  n_files = len(pot_files)

  # For all files (assumed to be images) in_directory
  for i, f in enumerate(pot_files):
    path_to_file = os.path.join(in_directory, f)

    # if the file is a file and not a folder
    if os.path.isfile(path_to_file):
      im = Image.open(path_to_file)

      # Resize
      #if im.size != resize_to:
      im = im.resize(resize_to)
      print('resized: {}/{}'.format(i, n_files))

      # Convert to Grayscale
      if convert_to_grayscale:
        im = im.convert('L')

      # Save to_directory
      im.save(os.path.join(out_directory, f))

def flattenImagesIntoArray(path_to_images, img_shape):
  # matrix of flattened images
  image_matrix = np.array([
    np.array(
      Image.open(
        os.path.join(path_to_images, img)
      )
    ).flatten() for img in os.listdir(path_to_images)
  ], 'f')

  data_array = image_matrix.astype('float32') / 255.
  data_array = np.reshape(data_array, (len(data_array), img_shape[0], img_shape[1], img_shape[2]))

  return data_array

def addNoiseToArray(data, amount, clip_min = 0, clip_max = 1.):
  # assumes data is a numpy array
  data_noisy = data + amount * np.random.normal(size = data.shape)
  return np.clip(data_noisy, 0.0, 1.0)

