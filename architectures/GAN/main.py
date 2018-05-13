import os, math
from datetime import datetime
from shutil import copytree

import cv2 as cv
import numpy as np

# Project Imports
import model as M
from model import config as C
from model import architecture
from callbacks import PlotLatentSpaceProgress, PlotLosses

from core.Generators import Generators
from core.processing import preProcessImages, flattenImagesIntoArray, addNoiseToArray
from core.ProjectStructure import TrainingSessionOutputDirectory

PATH_TO_THIS_DIR = os.path.dirname(__file__)

t = datetime.now()

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
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
N_DATA = sum([len(files) for r, d, files in os.walk(PATH_TO_DATASET)])

# Data Generators, used to load data in batches to save memory
# these are on the format (x, y) with input and expected output
# They also perform data augmentation on the fly: resize, greyscale, hue-shift, zoom etc.
data_generator = Generators(C).getDataGenerator(PATH_TO_DATASET)
#val_data_generator = Generators(C).getDataGenerator(PATH_TO_DATASET)

""" MODEL """
# GAN Instance
GAN = architecture.GAN()

# Load saved weights
if C.LOAD_SAVED_WEIGHTS:
  GAN.load(PATH_TO_LOAD_WEIGHTS)

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  GAN.printSummary()

""" TRAINING """
# Training Callback: Latent space progress
latent_space_progress = PlotLatentSpaceProgress(
  model = GAN.generator,
  config = C,
  tiling = C.LATENT_SPACE_TILING,
  img_size = C.PLOT_SIZE,
  max_dist_from_mean = 2,
  show_plot = C.SHOW_LATENT_PLOT,
  save_plot = C.SAVE_LATENT_PLOT,
  path_to_save_directory = PATHS.PATH_TO_EPOCH_PLOTS,
  save_name = M.NAME
)

# Training Callback: Plot Losses
plot_losses = PlotLosses(
  config = C,
  path_to_save_directory = PATHS.PATH_TO_LOSS_PLOTS
)

GAN.train(PATHS, N_DATA, data_generator, epoch_callbacks=[latent_space_progress, plot_losses], batch_callbacks=[plot_losses])

# Save model on completion
if C.SAVE_MODEL_WHEN_DONE:
  GAN.save(PATHS.PATH_TO_SAVED_MODELS)
