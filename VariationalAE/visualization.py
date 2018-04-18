import numpy as np
import matplotlib.pyplot as plt

def plotLatentSpace2D(model, tiling=15, tile_size=40):

  figure = np.zeros((tile_size * tiling, tile_size * tiling))

  # we will sample n points within [-15, 15] standard deviations
  grid_x = np.linspace(-15, 15, tiling)
  grid_y = np.linspace(-15, 15, tiling)

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z_sample = np.array([[xi, yi]])
      x_decoded = model.predict(z_sample)
      visualisation_image = x_decoded[0].reshape(tile_size, tile_size)
      figure[i * tile_size: (i + 1) * tile_size,
              j * tile_size: (j + 1) * tile_size] = visualisation_image


  #plt.clf()
  fig = plt.figure() #figsize=(10, 10)

  plt.title('Latent Space of z')
  plt.xlabel('z1')
  plt.ylabel('z2')
  plt.imshow(figure)

  #plt.savefig("./VariationalAE/progress_images/epoch{}-h1_size{}-h2_size{}-dropout{}.jpg".format(n_epoch, hidden_1_size, hidden_2_size, dropout_amount))
  #plt.show()
  plt.show(block=False)
