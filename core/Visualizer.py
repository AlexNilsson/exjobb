import os
import math
import numpy as np
import cv2 as cv

class Visualizer:
  def __init__(self, config):
    self.config = config

  def _px_to_display(self, images):
    """ Rescales imagedata from specified domain to [0, 255] uint """
    return self._rescale(images, toModel=False)

  def _px_to_model(self, images):
    """ Rescales imagedata from [0, 255] to specified domain """
    return self._rescale(images, toModel=True)

  def _rescale(self, images, toModel=False):
    """ Rescales imagedata from specified domain to [0, 255] uint """
    domain = self.config.GENERATED_IMAGE_RANGE
    
    if domain == (-1, 1):
      if toModel:
        images = 2 * images / 255 - 1
      else:
        images = (images + 1) * 255 / 2
        images = images.astype('uint8')

    elif domain == (0, 1):
      if toModel:
        images = images / 255
      else:
        images = images * 255
        images = images.astype('uint8')

    else:
      print('Unknown GENERATED_IMAGE_RANGE: {} in config.py'.format(domain))
      print('Please set this to either (-1, 1) or (0, 1) depending on activation function')
    
    return images


  def plotLatentSpace2D(self, model, tiling = 15, img_size = 720, max_dist_from_mean = 1, show_plot = True, channels = 3):
    C = self.config

    # Calculate appropriate tile size
    tile_size = math.ceil(img_size / tiling)
    # Create an empty canvas of appropriate size (This may be slightly larger than img_size due to px rounding)
    figure = np.zeros((tile_size * tiling, tile_size * tiling, channels))

    # sample n points within the distribution
    grid_x = np.linspace(-max_dist_from_mean, max_dist_from_mean, tiling)
    grid_y = np.linspace(-max_dist_from_mean, max_dist_from_mean, tiling)

    for i, yi in enumerate(grid_x):
      for j, xi in enumerate(grid_y):
        z_sample = np.ones(C.Z_LAYER_SIZE) * yi
        z_sample[0] = xi
        z_sample = np.array([z_sample])
        #z_sample = np.array([[xi, yi]])
        x_decoded = model.predict(z_sample)
        visualisation_image = x_decoded[0].reshape(C.IMG_SIZE, C.IMG_SIZE, channels)
        visualisation_image = cv.resize(visualisation_image, (tile_size, tile_size))
        visualisation_image = visualisation_image[...,::-1] #bgr to rgb
        figure[i * tile_size: (i + 1) * tile_size,
                j * tile_size: (j + 1) * tile_size, :] = visualisation_image

    # Resize figure down to img_size (in case it is larger due to tile_size px rounding)
    figure = cv.resize(figure, (img_size, img_size))

    if show_plot:
      cv.imshow('Latent Space 2D', figure)
      cv.waitKey(1)

    return figure

  def getLatentSpaceGrid(self, model, z_samples, img_size = 720, channels = 3):
    C = self.config

    n_samples = z_samples.shape[0]
    tiling = int(np.ceil(np.sqrt(n_samples)))

    # Calculate appropriate tile size
    tile_size = math.ceil(img_size / tiling)

    # Create an empty canvas of appropriate size (This may be slightly larger than img_size due to px rounding)
    figure = np.zeros((tile_size * tiling, tile_size * tiling, channels))

    x_samples = model.predict(z_samples)

    for i in range(tiling):
      for j in range(tiling):
        n = i * tiling + j

        if n >= n_samples:
          break
        else:
          x = x_samples[n]

          img = x.reshape(C.IMG_SIZE, C.IMG_SIZE, channels)
          img = cv.resize(img, (tile_size, tile_size))
          img = img[...,::-1] #bgr to rgb

          figure[i * tile_size: (i + 1) * tile_size,
                  j * tile_size: (j + 1) * tile_size, :] = img

    # Resize figure down to img_size (in case it is larger due to tile_size px rounding)
    figure = cv.resize(figure, (img_size, img_size))
    figure = self._px_to_display(figure)

    return figure

  def getReconstructionGrid(self, model, samples, img_size = 720, channels = 3):
    C = self.config

    n_samples = samples.shape[0]
    tiling = int(np.ceil(np.sqrt(n_samples*2))) #nr of tiles per row. *2 as we are adding the predicted ones every other row

    # Calculate appropriate tile size
    tile_size = math.ceil(img_size / tiling)

    # Create an empty canvas of appropriate size (This may be slightly larger than img_size due to px rounding)
    figure = np.zeros((tile_size * tiling, tile_size * tiling, channels))

    samples_recon = model.predict(samples)

    for i in [2*i for i in range(int(tiling/2))]: #0,2,4,
      for j in range(tiling):
        n = int((i/2) * tiling + j) #nr of samples processed

        if n >= n_samples:
          break
        else:
          x_recon = samples_recon[n]

          # reshape and convert sample to rgb
          img = x_recon.reshape(C.IMG_SIZE, C.IMG_SIZE, channels)
          img = cv.resize(img, (tile_size, tile_size))
          img = img[...,::-1] #bgr to rgb

          # add reconstructed img to canvas
          figure[i * tile_size: (i + 1) * tile_size,
                  j * tile_size: (j + 1) * tile_size, :] = img

          # add original img to the row below
          original_img = cv.resize(samples[n], (tile_size, tile_size))
          original_img = original_img[...,::-1] #bgr to rgb
          figure[(i+1) * tile_size: (i + 2) * tile_size,
                  j * tile_size: (j + 1) * tile_size, :] = original_img

    # Resize figure down to img_size (in case it is larger due to tile_size px rounding)
    figure = cv.resize(figure, (img_size, img_size))
    figure = self._px_to_model(figure)

    return figure


  def getImageForModelInput(self, path_to_img):
    """ Returns an image array in the correct format for the model defined in the config file """
    C = self.config
    img = cv.imread(path_to_img, C.COLOR_MODE == 'rgb')
    img = cv.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    img = self._px_to_model(img)
    img = img[...,::-1] #bgr to rgb
    return img

  def saveImg(self, data, output_dir, name):
    file_path = os.path.join(output_dir, name)
    cv.imwrite(file_path, data)
