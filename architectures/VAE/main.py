import os, math
from shutil import copytree

import cv2 as cv
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau, CSVLogger

# Project Imports
from core.Generators import Generators
from core.processing import preProcessImages, flattenImagesIntoArray, addNoiseToArray
from core.ProjectStructure import TrainingSessionOutputDirectory

import model as M
from model import config as C
from model import architecture
from callbacks import PlotLatentSpaceProgress, PlotLosses

PATH_TO_THIS_DIR = os.path.dirname(__file__)

""" OUTPUT DIRECTORIES """
PATHS = TrainingSessionOutputDirectory(PATH_TO_THIS_DIR, M.NAME)

# Backup current Architecture & Config file to this new working directory
copytree(os.path.join(PATH_TO_THIS_DIR, M.REL_PATH_TO_MODEL_DIR), os.path.join(PATHS.PATH_TO_OUT_DIRECTORY, M.NAME))

""" DATA TO LOAD """
PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

# Number of samples in the dataset
#N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

# Data Generators, used to load data in batches to save memory
# these are on the format (x, y) with input and expected output
# They also perform data augmentation on the fly: resize, greyscale, hue-shift, zoom etc.
#data_pair_generator = Generators(C).getDataPairGenerator(PATH_TO_DATASET)
#val_data_pair_generator = Generators(C).getDataPairGenerator(PATH_TO_DATASET)

""" MODEL """
# VAE Instance
VAE = architecture.VAE()

# Load saved weights
if C.LOAD_SAVED_WEIGHTS:
  VAE.load(PATH_TO_LOAD_WEIGHTS)

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  VAE.printSummary()

""" TRAINING """

# Training Callback: Latent space progress
latent_space_progress = PlotLatentSpaceProgress(
  model = VAE.decoder,
  config = C,
  tiling = C.LATENT_SPACE_TILING,
  img_size = C.PLOT_SIZE,
  max_dist_from_mean = 2,
  show_plot = C.SHOW_LATENT_PLOT,
  save_plot = C.SAVE_LATENT_PLOT,
  path_to_save_directory = PATHS.PATH_TO_EPOCH_PLOTS,
  save_name = M.NAME
)

# Training Callback: Weights checkpoint
weights_checkpoint_callback = ModelCheckpoint(
  filepath = os.path.join(PATHS.PATH_TO_SAVED_WEIGHTS, 'weights.hdf5'),
  verbose = 1,
  save_best_only = True,
  save_weights_only = True
)

# Training Callback: Plot Losses
plot_losses = PlotLosses(
  config = C,
  path_to_save_directory = PATHS.PATH_TO_LOSS_PLOTS
)

# Training Callback: Terminate if loss = NaN
terminate_on_nan = TerminateOnNaN()

# Training Callback: Tensorboard logs
tensorboard = TensorBoard(
  log_dir = PATHS.PATH_TO_LOGS_DIRECTORY,
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
  filename = os.path.join(PATHS.PATH_TO_LOGS_DIRECTORY, 'loss.log')
)

# Train Model
# Fit using data from generators

PATH_MNIST_DATA = os.path.join(PATH_TO_DATASET, 'train.csv')
mnist = pd.read_csv(PATH_MNIST_DATA).values

x_train = mnist[:, 1:].reshape(mnist.shape[0], 784,)
x_train = x_train.astype(float)
x_train /= 255.0
#x_train = x_train[:2000]

VAE.combined.fit(x_train, x_train, batch_size=C.BATCH_SIZE, epochs=C.EPOCHS, verbose=1, callbacks=[latent_space_progress, log_losses])

""" VAE.combined.fit(
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
) """

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  VAE.save(PATHS.PATH_TO_SAVED_MODELS)