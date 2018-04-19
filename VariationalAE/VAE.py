import os, math
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

# Project Imports
import config as C
import architecture
from callbacks import PlotLatentSpaceProgress
from generators import getDataPairGenerator

""" DATA """
PATH_TO_THIS_DIR = os.path.dirname(__file__)

PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, 'saved_weights')

PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

# Data Generators
data_pair_generator = getDataPairGenerator(PATH_TO_DATASET)
val_data_pair_generator = getDataPairGenerator(PATH_TO_DATASET)

""" MODEL """
# VAE Instance
VAE = architecture.VAE()

# Encoder Model
encoder = VAE.encoder

# Decoder Model
decoder = VAE.decoder

# VAE Model
vae = VAE.model

# Load saved weights
if C.LOAD_SAVED_WEIGHTS:
  before_weight_load = vae.get_weights()
  vae.load_weights(os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), by_name=False)
  after_weight_load = vae.get_weights()
  print('before_weight_load')
  print(before_weight_load)
  print('after_weight_load')
  print(after_weight_load)

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  print('vae summary:')
  vae.summary()
  print('encoder summary:')
  encoder.summary()
  print('decoder summary:')
  decoder.summary()

""" TRAINING """
# Optimizer
vae.compile(optimizer = 'adam', loss = VAE.loss_function)

# Training Callback: Latent space progress
latent_space_progress = PlotLatentSpaceProgress(
  model = decoder,
  tiling = C.LATENT_SPACE_TILING,
  tile_size = C.IMG_SIZE,
  zoom = 1, # Standard deviation, #TODO: Rename
  show_plot = True,
  save_plot = True,
  path_to_save_directory = os.path.join(PATH_TO_THIS_DIR, 'epoch_plots'),
  save_name = 'h1_size{}-h2_size{}-dropout{}'.format(C.HIDDEN_1_SIZE, C.HIDDEN_2_SIZE, C.DROPUT_AMOUNT)
)

# Training Callback: Weights checkpoint
weights_checkpoint_callback = ModelCheckpoint(
  filepath = os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'),
  verbose = 1,
  save_best_only = True,
  save_weights_only = True
)

# Train Model
vae.fit_generator(
  data_pair_generator,
  epochs = C.EPOCHS,
  steps_per_epoch = math.floor(N_DATA/C.BATCH_SIZE),
  validation_data = val_data_pair_generator,
  validation_steps = math.floor(N_DATA/C.BATCH_SIZE),
  shuffle = True,
  callbacks = [latent_space_progress],
  use_multiprocessing = False,
  verbose = C.TRAIN_VERBOSITY,
)

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  vae.save(os.path.join(PATH_TO_THIS_DIR, 'vae_model.h5'))
  vae.save_weights(os.path.join(PATH_TO_THIS_DIR,'vae_weights.h5'))
  decoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'decoder_weights.h5'))
  encoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'encoder_weights.h5'))
  decoder.save(os.path.join(PATH_TO_THIS_DIR,'decoder_model.h5'))
