""" Load / Save Model"""
LOAD_SAVED_WEIGHTS = False
LOAD_FROM_DIRECTORY = 'out/2018-04-20--13-10--model'

""" Data """
DATASET = 'windows'
IMG_SIZE = 64 # 320 # must be dividable by 8
COLOR_MODE = 'rgb' # 'grayscale' or 'rgb'

""" Data Augmentation """
SHUFFLE = True
ZOOM_RANGE = 0
CHANNEL_SHIFT_RANGE = 0
HORIZONTAL_FLIP = True

""" Architecture """
Z_LAYER_SIZE = 100
DROPUT_AMOUNT = 0.25
NOISE_FACTOR = 0

""" Training """
BATCH_SIZE = 59 # the trainingset must be dividable with batches_size
EPOCHS = 1000000
N_TRAIN_CRITIC = 5
GRADIENT_PENALTY_WEIGHT = 10
SAVE_WEIGHTS_FREQ = 5 # How often to save weights (n epochs)
SAVE_MODEL_ON_COMPLETION = False

""" Debug """
PRINT_MODEL_SUMMARY = True
TRAIN_VERBOSITY = 1 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

PLOT_LATENT_SPACE_EVERY = 10 #How often to plot the latent space, 1= every epoch, 10 every 10th etc.
PLOT_LOSS_EVERY = 50 #How often to plot the loss plot, 1= every epoch, 10 every 10th etc.
PLOT_SIZE = 720 #px
LATENT_SPACE_TILING = 15
SHOW_LATENT_PLOT = False
SAVE_LATENT_PLOT = True
