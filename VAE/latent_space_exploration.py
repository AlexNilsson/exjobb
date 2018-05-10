import os
import cv2 as cv
import numpy as np
from datetime import datetime

# Project Imports
import model as M
from model import config as C
from model import architecture
from visualization import saveImg, getLatentSpaceGrid

PATH_TO_THIS_DIR = os.path.dirname(__file__)

t = datetime.now()

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
THIS_ID = "lsexp_{}--{}-{}-{}--{}-{}-{}".format(M.NAME, t.year, t.month, t.day, t.hour, t.minute, t.second)
PATH_TO_OUT_DIRECTORY = os.path.join(PATH_TO_THIS_DIR, 'out_exp', THIS_ID)
os.makedirs(PATH_TO_OUT_DIRECTORY, exist_ok = True)

PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

""" MODEL """
# VAE Instance
VAE = architecture.VAE()

# Encoder Model
encoder = VAE.encoder

# Decoder Model
decoder = VAE.decoder

# VAE Model
vae = VAE.model

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  print('vae summary:')
  vae.summary()
  print('encoder summary:')
  encoder.summary()
  print('decoder summary:')
  decoder.summary()


""" LOAD SAVED WEIGHTS """
if True:
  try:
    before_weight_load = vae.get_weights()
    vae.load_weights(os.path.join(PATH_TO_LOAD_WEIGHTS, 'weight.hdf5'), by_name=False)
    after_weight_load = vae.get_weights()

  except Exception as e:
    print('Failed to load stored weights..')
    print(str(e))

  else:
    if np.array_equal(before_weight_load, after_weight_load):
      print('Loaded weights are identical?')
    else:
      print('Weights loaded successfully!')

""" SAMPLE POINTS IN LATENT SPACE """
N_SAMPLES_PER_IMG = 36
N_IMAGES = 300

# Print Grids of randomly sampled points
img_size = int(np.ceil(np.sqrt(N_SAMPLES_PER_IMG)) * C.IMG_SIZE)

""" PLOT RANDOM LATENT SPACE POINTS"""
if False:
  for i in range(N_IMAGES):
    z_samples = np.random.normal(0, i + np.abs(np.random.normal(0,3)), (N_SAMPLES_PER_IMG, C.Z_LAYER_SIZE))

    # Decode vectors and get image
    img = getLatentSpaceGrid(decoder, z_samples, img_size = img_size)

    # Save img
    saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(i))


""" PLOT 1 HOT LATENT VECTORS """
if False:
  # Generate 1-hot vectors
  z_samples = np.eye(C.Z_LAYER_SIZE)

  # Decode vectors and get image
  img = getLatentSpaceGrid(decoder, z_samples, img_size = img_size)

  # Save img
  saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration_1hot_vectors-{}.jpg'.format(1))


""" PLOT RANDOM LATENT SPACE INTERPOLATION"""
if True:
  for i in range(N_IMAGES):
    z_samples = np.random.normal(0, 2, (2, C.Z_LAYER_SIZE))
    z_1, z_2 = z_samples[0], z_samples[1]

    x = z_2 - z_1

    z_samples = np.zeros((N_SAMPLES_PER_IMG, C.Z_LAYER_SIZE))

    for j in range(N_SAMPLES_PER_IMG):
      z_samples[j] = z_1 + x * (j+1)/N_SAMPLES_PER_IMG

      # Decode vectors and get image
      img = getLatentSpaceGrid(decoder, z_samples, img_size = img_size)

      # Save img
      saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(i))