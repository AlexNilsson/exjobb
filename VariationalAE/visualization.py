import numpy as np
import cv2 as cv

def plotLatentSpace2D(model, tiling=15, tile_size=40, zoom=1, show_plot=True):

  figure = np.zeros((tile_size * tiling, tile_size * tiling))

  # we will sample n points within [-15, 15] standard deviations
  grid_x = np.linspace(-zoom, zoom, tiling)
  grid_y = np.linspace(-zoom, zoom, tiling)

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z_sample = np.array([[xi, yi]])
      x_decoded = model.predict(z_sample)
      visualisation_image = x_decoded[0].reshape(tile_size, tile_size)
      figure[i * tile_size: (i + 1) * tile_size,
              j * tile_size: (j + 1) * tile_size] = visualisation_image

  if show_plot:
    cv.imshow('Latent Space 2D', figure)
    cv.waitKey(1)

  return figure
