import math
import numpy as np
import cv2 as cv
from PIL import Image

from model import config as C

def plotLatentSpace2D(model, tiling = 15, img_size = 720, max_dist_from_mean = 1, show_plot = True, channels = 3):

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
