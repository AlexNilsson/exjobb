import os, math
from datetime import datetime
from shutil import copytree

import cv2 as cv
import numpy as np

from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau, CSVLogger

# Project Imports
import model as M
from model import config as C
from model import architecture

from callbacks import PlotLatentSpaceProgress, PlotLosses
from core.Generators import Generators
from core.processing import preProcessImages, flattenImagesIntoArray, addNoiseToArray

PATH_TO_THIS_DIR = os.path.dirname(__file__)

t = datetime.now()

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
THIS_ID = "{}--{}-{}-{}--{}-{}-{}".format(M.NAME, t.year, t.month, t.day, t.hour, t.minute, t.second)
PATH_TO_OUT_DIRECTORY = os.path.join(PATH_TO_THIS_DIR, 'out', THIS_ID)
PATH_TO_LOGS_DIRECTORY = os.path.join(PATH_TO_OUT_DIRECTORY, 'logs')

# sub folders
PATH_TO_EPOCH_PLOTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'epoch_plots')
PATH_TO_LOSS_PLOTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'loss_plots')
PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')
PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_weights')
PATH_TO_SAVED_MODELS = os.path.join(PATH_TO_OUT_DIRECTORY, 'saved_models')

PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

# create folders if they do not already exist
os.makedirs(PATH_TO_OUT_DIRECTORY, exist_ok = True)
os.makedirs(PATH_TO_LOGS_DIRECTORY, exist_ok = True)
os.makedirs(PATH_TO_EPOCH_PLOTS, exist_ok = True)
os.makedirs(PATH_TO_LOSS_PLOTS, exist_ok = True)
os.makedirs(PATH_TO_SAVED_WEIGHTS, exist_ok = True)
os.makedirs(PATH_TO_SAVED_MODELS, exist_ok = True)

# Backup current Architecture & Config file to this new working directory
copytree(os.path.join(PATH_TO_THIS_DIR, M.REL_PATH_TO_MODEL_DIR), os.path.join(PATH_TO_OUT_DIRECTORY, M.NAME))

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

# Number of samples in the dataset
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

# Data Generators, used to load data in batches to save memory
# these are on the format (x, y) with input and expected output
# They also perform data augmentation on the fly: resize, greyscale, hue-shift, zoom etc.
data_pair_generator = Generators(C).getDataPairGenerator(PATH_TO_DATASET)
val_data_pair_generator = Generators(C).getDataPairGenerator(PATH_TO_DATASET)

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
  vae.load_weights(os.path.join(PATH_TO_LOAD_WEIGHTS, 'weight.hdf5'), by_name=False)
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
vae.compile(optimizer = VAE.optimizer, loss = VAE.loss_function)

# Training Callback: Latent space progress
latent_space_progress = PlotLatentSpaceProgress(
  model = decoder,
  config = C,
  tiling = C.LATENT_SPACE_TILING,
  img_size = C.PLOT_SIZE,
  max_dist_from_mean = 2,
  show_plot = C.SHOW_LATENT_PLOT,
  save_plot = C.SAVE_LATENT_PLOT,
  path_to_save_directory = PATH_TO_EPOCH_PLOTS,
  save_name = M.NAME
)

# Training Callback: Weights checkpoint
weights_checkpoint_callback = ModelCheckpoint(
  filepath = os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'),
  verbose = 1,
  save_best_only = True,
  save_weights_only = True
)

# Training Callback: Plot Losses
plot_losses = PlotLosses(
  config = C,
  path_to_save_directory = PATH_TO_LOSS_PLOTS
)

# Training Callback: Terminate if loss = NaN
terminate_on_nan = TerminateOnNaN()

# Training Callback: Tensorboard logs
tensorboard = TensorBoard(
  log_dir = PATH_TO_LOGS_DIRECTORY,
  histogram_freq = 0,
  batch_size = 32,
  write_graph=True,
  write_grads=False,
  write_images=False,
  embeddings_freq=0,
  embeddings_layer_names=None,
  embeddings_metadata=None
)

# Training Callback: Reduce learning rate upon reaching a plateau
reduce_lr_on_plateau = ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.1,
  patience=100,
  verbose=1,
)

# Training Callback: log losses
log_losses = CSVLogger(
  filename = os.path.join(PATH_TO_LOGS_DIRECTORY, 'loss.log')
)

# Train Model
# Fit using data from generators
vae.fit_generator(
  data_pair_generator,
  epochs = C.EPOCHS,
  steps_per_epoch = math.floor(N_DATA/C.BATCH_SIZE),
  validation_data = val_data_pair_generator,
  validation_steps = math.ceil((N_DATA/C.BATCH_SIZE)/10),
  shuffle = C.SHUFFLE,
  callbacks = [
    terminate_on_nan,
    reduce_lr_on_plateau,
    latent_space_progress,
    weights_checkpoint_callback,
    log_losses
    ],
  use_multiprocessing = False,
  verbose = C.TRAIN_VERBOSITY,
  initial_epoch = C.INIT_EPOCH if C.LOAD_SAVED_WEIGHTS else 0
)

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  vae.save(os.path.join(PATH_TO_SAVED_MODELS, 'vae_model.h5'))
  vae.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'vae_weights.h5'))
  decoder.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'decoder_weights.h5'))
  encoder.save_weights(os.path.join(PATH_TO_SAVED_MODELS,'encoder_weights.h5'))
  decoder.save(os.path.join(PATH_TO_SAVED_MODELS,'decoder_model.h5'))
