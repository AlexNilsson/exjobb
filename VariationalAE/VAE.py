import os, math
import matplotlib.pyplot as plt
from datetime.datetime import now
from shutil import copyfile

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

t = now()

PATH_TO_THIS_DIR = os.path.dirname(__file__)

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
THIS_ID = "{}-{}-{}--{}-{}-{}--{}".format(t.year, t.month, t.day, t.hour, t.minute, t.second, C.NAME)
PATH_TO_OUT_DIRECTORY = os.path.join(PATH_TO_THIS_DIR, 'out', THIS_ID)

# sub folders
PATH_TO_EPOCH_PLOTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'epoch_plots')
PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_weights')
PATH_TO_SAVED_MODELS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_models')

# create folders if they do not already exist
if not os.path.exists(PATH_TO_OUT_DIRECTORY): os.mkdir(PATH_TO_OUT_DIRECTORY)
if not os.path.exists(PATH_TO_OUT_DIRECTORY): os.mkdir(PATH_TO_EPOCH_PLOTS)
if not os.path.exists(PATH_TO_OUT_DIRECTORY): os.mkdir(PATH_TO_SAVED_WEIGHTS)
if not os.path.exists(PATH_TO_OUT_DIRECTORY): os.mkdir(PATH_TO_SAVED_MODELS)

# Backup current Architecture & Config file to this new working directory
copyfile(os.path.join(PATH_TO_THIS_DIR, 'architecture.py'), PATH_TO_OUT_DIRECTORY)
copyfile(os.path.join(PATH_TO_THIS_DIR, 'config.py'), PATH_TO_OUT_DIRECTORY)

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

# Number of samples in the dataset
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

if C.USE_GENERATORS:
  # Data Generators, used to load data in batches to save memory
  # these are on the format (x, y) with input and expected output
  data_pair_generator = getDataPairGenerator(PATH_TO_DATASET)
  val_data_pair_generator = getDataPairGenerator(PATH_TO_DATASET)
else:


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
  img_size = 720,
  max_dist_from_mean = 2,
  show_plot = C.SHOW_LATENT_PLOT,
  save_plot = C.SAVE_LATENT_PLOT,
  path_to_save_directory = PATH_TO_EPOCH_PLOTS,
  save_name = C.NAME
)

# Training Callback: Weights checkpoint
weights_checkpoint_callback = ModelCheckpoint(
  filepath = os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'),
  verbose = 1,
  save_best_only = True,
  save_weights_only = True
)

# Train Model
if C.USE_GENERATORS:
  # Fit using data from generators
  vae.fit_generator(
    data_pair_generator,
    epochs = C.EPOCHS,
    steps_per_epoch = math.floor(N_DATA/C.BATCH_SIZE),
    validation_data = val_data_pair_generator,
    validation_steps = math.ceil((N_DATA/C.BATCH_SIZE)/10),
    shuffle = C.SHUFFLE,
    callbacks = [latent_space_progress, weights_checkpoint_callback],
    use_multiprocessing = False,
    verbose = C.TRAIN_VERBOSITY
  )
else:
  # Fit using data already in memory
  vae.fit(
    x_train, x_train,
    shuffle = C.SHUFFLE,
    epochs = C.EPOCHS,
    batch_size = C.BATCH_SIZE,
    validation_data = (x_test, x_test)
  )

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  vae.save(os.path.join(PATH_TO_SAVED_MODELS, 'vae_model.h5'))
  vae.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'vae_weights.h5'))
  decoder.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'decoder_weights.h5'))
  encoder.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'encoder_weights.h5'))
  decoder.save(os.path.join(PATH_TO_SAVED_MODELS,'decoder_model.h5'))
