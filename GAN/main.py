import os, math
from datetime import datetime
from shutil import copytree

import cv2 as cv
import numpy as np
from PIL import Image

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Project Imports
import model as M
from model import config as C
from model import architecture

from callbacks import PlotLatentSpaceProgress, PlotLosses
from generators import getDataGenerator
from processing import preProcessImages, flattenImagesIntoArray, addNoiseToArray

PATH_TO_THIS_DIR = os.path.dirname(__file__)

t = datetime.now()

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
THIS_ID = "{}--{}-{}-{}--{}-{}-{}".format(M.NAME, t.year, t.month, t.day, t.hour, t.minute, t.second)
PATH_TO_OUT_DIRECTORY = os.path.join(PATH_TO_THIS_DIR, 'out', THIS_ID)

# sub folders
PATH_TO_EPOCH_PLOTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'epoch_plots')
PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_weights')
PATH_TO_SAVED_MODELS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_models')

# create folders if they do not already exist
os.makedirs(PATH_TO_OUT_DIRECTORY, exist_ok = True)
os.makedirs(PATH_TO_EPOCH_PLOTS, exist_ok = True)
os.makedirs(PATH_TO_SAVED_WEIGHTS, exist_ok = True)
os.makedirs(PATH_TO_SAVED_MODELS, exist_ok = True)

# Backup current Architecture & Config file to this new working directory
copytree(os.path.join(PATH_TO_THIS_DIR, M.REL_PATH_TO_MODEL_DIR), os.path.join(PATH_TO_OUT_DIRECTORY, M.NAME))

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

# Number of samples in the dataset
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

if C.USE_GENERATORS:
  # Data Generators, used to load data in batches to save memory
  # these are on the format (x, y) with input and expected output
  # They also perform data augmentation on the fly: resize, greyscale, hue-shift, zoom etc.
  data_generator = getDataGenerator(PATH_TO_DATASET)
  #val_data_generator = getDataGenerator(PATH_TO_DATASET)

else:
  # Preparses the trainingset to a new folder on disc and
  # loads everything into memory in one batch
  PATH_TO_PROCESSED_DATASET = os.path.join(PATH_TO_DATASET, 'processed')
  if not os.path.exists(PATH_TO_PROCESSED_DATASET): os.mkdir(PATH_TO_PROCESSED_DATASET)

  # preprocess data to new directory
  preProcessImages(PATH_TO_DATASET, PATH_TO_PROCESSED_DATASET,
    convert_to_grayscale = C.COLOR_MODE == 'greyscale',
    resize_to = (C.IMG_SIZE, C.IMG_SIZE)
  )

  # Flatten and load all data into an array
  channels = 3 if C.COLOR_MODE == 'rgb' else 1
  img_shape = (C.IMG_SIZE, C.IMG_SIZE, channels)
  x_train = flattenImagesIntoArray(PATH_TO_PROCESSED_DATASET, img_shape)

  x_train_ref = x_train

  # Add noise?
  if C.NOISE_FACTOR > 0:
    x_train = addNoiseToArray(x_train, C.NOISE_FACTOR)

  x_test, x_test_ref = x_train, x_train_ref

""" MODEL """
# GAN Instance
GAN = architecture.GAN()

# Load saved weights
if C.LOAD_SAVED_WEIGHTS:
  before_weight_load = gan.get_weights()
  gan.load_weights(os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), by_name=False)
  after_weight_load = gan.get_weights()
  print('before_weight_load')
  print(before_weight_load)
  print('after_weight_load')
  print(after_weight_load)

""" TRAINING """
# Optimizer
optimizer = Adam(0.0002, 0.5)

# compile Generator model
GAN.generator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

# compile Discriminator model
GAN.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

# compile Combined model, with frozen discriminator
GAN.combined.get_layer(name='Discriminator').trainable = False
GAN.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  print('gan summary:')
  GAN.combined.summary()
  print('generator summary:')
  GAN.generator.summary()
  print('discriminator summary:')
  GAN.discriminator.summary()

# Training Callback: Latent space progress
latent_space_progress = PlotLatentSpaceProgress(
  model = GAN.generator,
  tiling = C.LATENT_SPACE_TILING,
  img_size = C.PLOT_SIZE,
  max_dist_from_mean = 2,
  show_plot = C.SHOW_LATENT_PLOT,
  save_plot = C.SAVE_LATENT_PLOT,
  path_to_save_directory = PATH_TO_EPOCH_PLOTS,
  save_name = M.NAME
)

# Training Callback: Plot Losses
plot_losses = PlotLosses()

DISC_TRAIN_THRESH = -math.log(0.5) # ~cross entropy loss at 50% correct classification

batches = math.floor(N_DATA/C.BATCH_SIZE)


d_loss_epoch = math.inf
g_loss_epoch = math.inf

for epoch in range(1, C.EPOCHS + 1):

  train_d = d_loss_epoch > DISC_TRAIN_THRESH or ( d_loss_epoch < DISC_TRAIN_THRESH and g_loss_epoch < DISC_TRAIN_THRESH )
  train_g = g_loss_epoch > DISC_TRAIN_THRESH or ( d_loss_epoch < DISC_TRAIN_THRESH and g_loss_epoch < DISC_TRAIN_THRESH )

  d_loss_epoch = []
  g_loss_epoch = []

  latent_space_progress.on_epoch_begin(epoch, None)

  for bath_idx in range(data_generator.__len__()):
    batch = data_generator.__getitem__(bath_idx)

    if True:
      """ Train Discriminator """
      # Select a random half batch of images
      real_data = batch[np.random.randint(0, int(C.BATCH_SIZE), int(C.BATCH_SIZE/2))]

      # Sample noise and generate a half batch of new images
      #flat_img_length = batch.shape[1]
      noise = np.random.normal(0, 1, (int(C.BATCH_SIZE/2), int(C.Z_LAYER_SIZE)))
      fake_data = GAN.generator.predict(noise)

      # Train the discriminator (real classified as ones and generated as zeros)
      d_loss_real = GAN.discriminator.train_on_batch(real_data, np.ones((int(C.BATCH_SIZE/2), 1)))
      d_loss_fake = GAN.discriminator.train_on_batch(fake_data, np.zeros((int(C.BATCH_SIZE/2), 1)))
      d_loss = np.mean([d_loss_real, d_loss_fake])
      d_loss_epoch.append(d_loss)

    if True:
      """ Train Generator """
      # Sample generator input
      noise = np.random.normal(0, 1, (int(C.BATCH_SIZE), int(C.Z_LAYER_SIZE)))

      # Train the generator to fool the discriminator, e.g. classify these images as real (1)
      # The discriminator model is frozen in this stage but its gradient is still used to guide the generator
      g_loss = GAN.combined.train_on_batch(noise, np.ones((C.BATCH_SIZE, 1)))
      g_loss_epoch.append(g_loss)

    # Plot the progress
    print("Epoch:{} Batch:{}/{} [D loss: {}] [G loss: {}]".format(epoch, bath_idx+1, batches, d_loss, g_loss))

  d_loss_epoch = np.mean(d_loss_epoch)
  g_loss_epoch = np.mean(g_loss_epoch)

  latent_space_progress.on_epoch_end(epoch, None)
  plot_losses.on_epoch_end(epoch, {'d_loss': d_loss_epoch, 'g_loss': g_loss_epoch})
  data_generator.on_epoch_end()

  if epoch % C.SAVE_WEIGHTS_FREQ == 0:
    GAN.combined.save_weights(os.path.join(PATH_TO_SAVED_WEIGHTS,'combined_weight.hdf5'))

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  GAN.combined.save(os.path.join(PATH_TO_SAVED_MODELS, 'gan_model.h5'))
  GAN.combined.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'gan_weights.h5'))

  GAN.generator.save(os.path.join(PATH_TO_SAVED_MODELS, 'generator_model.h5'))
  GAN.generator.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'generator_weights.h5'))

  GAN.discriminator.save(os.path.join(PATH_TO_SAVED_MODELS,'discriminator_model.h5'))
  GAN.discriminator.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'discriminator_weights.h5'))
