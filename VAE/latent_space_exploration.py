import os
import cv2 as cv
import numpy as np
from datetime import datetime

from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf

# Project Imports
import model as M
from model import config as C
from model import architecture
from visualization import saveImg, getLatentSpaceGrid, getReconstructionGrid

PATH_TO_THIS_DIR = os.path.dirname(__file__)

t = datetime.now()

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
THIS_ID = "lsexp_{}--{}-{}-{}--{}-{}-{}".format(M.NAME, t.year, t.month, t.day, t.hour, t.minute, t.second)
PATH_TO_OUT_DIRECTORY = os.path.join(PATH_TO_THIS_DIR, 'out_exp', THIS_ID)
os.makedirs(PATH_TO_OUT_DIRECTORY, exist_ok = True)

PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

# Number of samples in the dataset
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

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
if False:
  for i in range(N_IMAGES):
    z_samples = np.random.normal(0, 2, (2, C.Z_LAYER_SIZE))
    z_1, z_2 = z_samples[0], z_samples[1]

    x = z_2 - z_1

    z_samples = np.zeros((N_SAMPLES_PER_IMG, C.Z_LAYER_SIZE))

    for j in range(N_SAMPLES_PER_IMG):
      # Vector interpolation
      z_samples[j] = z_1 + x * (j+1)/N_SAMPLES_PER_IMG

      # Decode vectors and get image
      img = getLatentSpaceGrid(decoder, z_samples, img_size = img_size)

      # Save img
      saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(i))


""" RECONSTRUCTION OF REAL DATA """
if False:
  N_SAMPLES = 200

  np.random.randint(0, N_DATA, N_SAMPLES)

  data_generator = ImageDataGenerator().flow_from_directory(
    PATH_TO_DATASET,
    target_size = (C.IMG_SIZE, C.IMG_SIZE),
    batch_size = N_SAMPLES,
    class_mode = None,
    color_mode = C.COLOR_MODE
  )

  # Get one batch from generator
  for batch in data_generator:
    break
  
  print(batch.shape)
  batch = batch / 255

  reconstructed_data = vae.predict(batch)

  # Decode vectors and get image
  img = getReconstructionGrid(vae, batch, img_size = 2048)
  cv.imshow('plot', img)
  cv.waitKey(0)

  # Save img
  i=1
  saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(i))
  print(reconstructed_data.shape)


""" PLOT LATENT SPACE INTERPOLATION BASED ON DATA """
if True:
  N_SAMPLES = 2

  np.random.randint(0, N_DATA, N_SAMPLES)

  data_generator = ImageDataGenerator().flow_from_directory(
    PATH_TO_DATASET,
    target_size = (C.IMG_SIZE, C.IMG_SIZE),
    batch_size = N_SAMPLES,
    class_mode = None,
    color_mode = C.COLOR_MODE
  )

  # Get one batch from generator
  for batch in data_generator:
    break

  batch = batch / 255
  mu, log_sigma  = encoder.predict(batch)

  # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
  eps = K.random_normal(shape=(K.shape(mu)[0], C.Z_LAYER_SIZE), mean=0, stddev=1)
  # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
  z = mu + K.exp(log_sigma) * eps


  print("z-shape: {}".format(z.shape))
  z_1 = tf.to_float(z[0])
  z_2 = tf.to_float(z[1])
  print(z_1)
  print(z_2)

  x = z_2 - z_1
  print(x)

  z_samples = np.zeros((N_SAMPLES_PER_IMG, C.Z_LAYER_SIZE))

  for i in range(N_SAMPLES_PER_IMG):
    # Vector interpolation
    z_samples[i] = z_1 + x * (i+1)/N_SAMPLES_PER_IMG

    # Decode vectors and get image
    img = getLatentSpaceGrid(decoder, z_samples, img_size = img_size)

    # Save img
    saveImg(img, PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(1))