""" Load / Save Model"""
LOAD_SAVED_WEIGHTS = False
LOAD_FROM_DIRECTORY = 'out/2018-04-20--13-10--model'
SAVE_MODEL_WHEN_DONE = False

""" Data """
DATASET = 'windows'
IMG_SIZE = 64 # 320 # must be dividable by 8
COLOR_MODE = 'rgb' # 'grayscale' or 'rgb'
USE_GENERATORS = True

""" Data Augmentation """
SHUFFLE = True
# Only applicable if USE_GENERATORS = True
ZOOM_RANGE = 0.2
CHANNEL_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True

""" Architecture """
Z_LAYER_SIZE = 10
DROPUT_AMOUNT = 0.4
NOISE_FACTOR = 0

""" Training """
BATCH_SIZE = 913 # the trainingset must be dividable with batches_size
EPOCHS = 10000
KL_FACTOR = 0.5 # 1 = only KL, 0 = only Reconstruction Loss

""" Debug """
PRINT_MODEL_SUMMARY = True
TRAIN_VERBOSITY = 1 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

PLOT_LATENT_SPACE_EVERY = 5 #How often to plot the latent space, 1= every epoch, 10 every 10th etc.
PLOT_SIZE = 720 #px
LATENT_SPACE_TILING = 15
SHOW_LATENT_PLOT = False
SAVE_LATENT_PLOT = True
