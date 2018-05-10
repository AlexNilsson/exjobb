""" Load / Save Model"""
LOAD_SAVED_WEIGHTS = True
LOAD_FROM_DIRECTORY = 'out/VAE_Radford--2018-5-4--18-22-35'
INIT_EPOCH = 7500
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
Z_LAYER_SIZE = 100
DROPUT_AMOUNT = 0.4
NOISE_FACTOR = 0

""" Training """
LEARNING_RATE = 1e-06
BATCH_SIZE = 415 # the trainingset must be dividable with batches_size
EPOCHS = 20000
KL_FACTOR = 0.5 # 1 = only KL, 0 = only Reconstruction Loss

""" Debug """
PRINT_MODEL_SUMMARY = True
TRAIN_VERBOSITY = 1 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

PLOT_LATENT_SPACE_EVERY = 5 #How often to plot the latent space, 1= every epoch, 10 every 10th etc.
SAVE_LOSS_PLOT_FREQ = 50 # How often to save the the loss plot, 1= every epoch, 10 every 10th etc.
PLOT_SIZE = 720 #px
LATENT_SPACE_TILING = 15
SHOW_LATENT_PLOT = False
SAVE_LATENT_PLOT = True
