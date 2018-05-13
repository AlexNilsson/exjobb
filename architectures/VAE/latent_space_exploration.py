import os
import cv2 as cv
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# Project Imports
from core.Visualizer import Visualizer
from core.ProjectStructure import InferenceSessionOutputDirectory
from core.Math import vectorLerp, vectorSlerp

import model as M
from model import config as C
from model import architecture

visualizer = Visualizer(C)

PATH_TO_THIS_DIR = os.path.dirname(__file__)

""" OUTPUT DIRECTORIES """
# Create a new output directory for this training session with a unique name
PATHS = InferenceSessionOutputDirectory(PATH_TO_THIS_DIR, M.NAME)

""" DATA TO LOAD """
PATH_TO_LOAD_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, C.LOAD_FROM_DIRECTORY, 'saved_weights')

""" DATA """
# Path to dataset
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, C.DATASET)

""" MODEL """
# VAE Instance
VAE = architecture.VAE()

# Print model summary
if C.PRINT_MODEL_SUMMARY:
  VAE.printSummary()

""" LOAD SAVED WEIGHTS """
VAE.load(PATH_TO_LOAD_WEIGHTS)


def getVectorInterpolations(a, b, steps, interpolation_function):
  """ Creates a list of interpolation steps between two vectors """
  interpols = []
  for i in range(steps):
    interpols.append(
      interpolation_function(a, b, (i + 1) / steps )
    )
  return interpols


""" Returns the latent vectors for the provided images """
def encode_images(images, encoder, std = 0):
  # The std is the uncertainty to use when sampling from the learnt distribution

  # encoded distributions
  mu, log_sigma = encoder.predict(images)

  # random variable
  eps = np.random.normal(0, std, (mu.shape[0], C.Z_LAYER_SIZE))

  # z encodings
  z_samples = mu + np.exp(log_sigma) * eps

  return z_samples

""" GET RANDOM LATENT VECTORS """
def get_random_latent_space_points(n_points, std=1):
  for i in range(N_IMAGES):

    # Sample random latent vectors
    z_samples = np.random.normal(0, i + np.abs(np.random.normal(0, std)), (n_points, C.Z_LAYER_SIZE))

    return z_samples

""" GET 1 HOT LATENT VECTORS """
def get_1hot_latent_vectors():

  # Generate 1-hot vectors
  z_samples = np.eye(C.Z_LAYER_SIZE)

  return z_samples

""" GET RANDOM LATENT SPACE INTERPOLATION BETWEEN 2 POINTS """
def get_random_latent_interpolation(n_samples, std=1):

  # sample z vectors from normal distribution
  z_1, z_2 = np.random.normal(0, std, (2, C.Z_LAYER_SIZE))

  # Interpolation
  z_lerps = np.array(
    getVectorInterpolations(z_1, z_2, n_samples, vectorLerp)
  )

  return z_lerps

""" GET INTERPOLATION BETWEEN 2 TRAINED DATA POINTS """
def get_latent_lerp_of_train_data(encoder, data_samples, n_samples):
  # data_samples should be a list of 2 images to lerp between

  z_1, z_2 = encode_images(data_samples, encoder)

  # Interpolation
  z_lerps = np.array(
    getVectorInterpolations(z_1, z_2, n_samples, vectorLerp)
  )

  return z_lerps


""" RECONSTRUCTION OF REAL DATA """
def get_reconstructed_train_data(vae, path_to_dataset, n_samples, img_size = 720):
  # Plocks random data samples from the dataset
  data_generator = ImageDataGenerator().flow_from_directory(
    path_to_dataset,
    target_size = (C.IMG_SIZE, C.IMG_SIZE),
    batch_size = n_samples,
    class_mode = None,
    color_mode = C.COLOR_MODE
  )

  # Get one batch from generator
  for batch in data_generator: break
  batch = batch / 255

  # Decode vectors and get image
  return visualizer.getReconstructionGrid(vae, batch, img_size = img_size)

""" SAMPLE POINTS IN LATENT SPACE """
if True:

  N_IMAGES = 1
  N_SAMPLES_PER_IMG = 36

  PATH_TO_DATASET_DATA = os.path.join(PATH_TO_DATASET, os.listdir(PATH_TO_DATASET)[0])

  # Print Grids of randomly sampled points
  img_size = int(np.ceil(np.sqrt(N_SAMPLES_PER_IMG)) * C.IMG_SIZE)

  # Data generator to load data from the dataset
  data_generator = ImageDataGenerator().flow_from_directory(
    PATH_TO_DATASET,
    target_size = (C.IMG_SIZE, C.IMG_SIZE),
    batch_size = 2,
    class_mode = None,
    color_mode = C.COLOR_MODE
  )

  # Loop to output multiple images using the same function
  # Select what type of plot to ouput by uncommenting appropriate rows below
  for i in range(N_IMAGES):
    """ GET RANDOM LATENT VECTORS """
    #z_vectors = get_random_latent_space_points(N_SAMPLES_PER_IMG, std=2)

    """ GET 1 HOT LATENT VECTORS """
    #z_vectors = get_1hot_latent_vectors()

    """ GET RANDOM LATENT SPACE INTERPOLATION BETWEEN 2 POINTS """
    #z_vectors = get_random_latent_interpolation(N_SAMPLES_PER_IMG, std=2)

    """ GET INTERPOLATION BETWEEN TWO TRAINED DATA POINTS """
    #data_samples = data_generator.__getitem__(i) / 255
    #z_vectors = get_latent_lerp_of_train_data(encoder, data_samples, N_SAMPLES_PER_IMG)

    """ LERP: SQUARE WINDOW TO CIRCULAR WINDOW """
    if False:
      # Square wooden window, no mullions, white background
      img_1 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '1tqyx7sdcj.jpg'))

      # Round wooden window, no mullions, white background
      img_2 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '1xlmtz85qp.jpg'))

      z_vectors = get_latent_lerp_of_train_data(VAE.encoder, np.array([img_1, img_2]), N_SAMPLES_PER_IMG)

    """ LERP: SQUARE WINDOW TO CIRCULAR WINDOW """
    if False:
      # Square wooden window, no mullions, white background
      img_1 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '1tqyx7sdcj.jpg'))

      # Octagon wooden window, no mullion, white background
      img_2 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '61opbxtz7f.jpg'))

      # Round wooden window, no mullions, white background
      img_3 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '1xlmtz85qp.jpg'))

      img1_to_img2 = get_latent_lerp_of_train_data(VAE.encoder, np.array([img_1, img_2]), N_SAMPLES_PER_IMG)
      img2_to_img3 = get_latent_lerp_of_train_data(VAE.encoder, np.array([img_2, img_3]), N_SAMPLES_PER_IMG)

      z_vectors = np.append(img1_to_img2, img2_to_img3, axis=0)

    """ LATENT SPACE ARITHMIC: Mullions """
    if False:
      # Square wooden window, no mullions, white background
      img_1 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, 'bhoaezom15.jpg'))

      # Square wooden window, with mullions, white background
      img_2 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, 'bhvrj5prhv.jpg'))

      # Round wooden window, no mullions, white background
      img_3 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, '1xlmtz85qp.jpg'))

      # Round window, colorful glass, mullions, white background
      img_4 = visualizer.getImageForModelInput( os.path.join(PATH_TO_DATASET_DATA, 'cm6siputnb.jpg'))

      z_1, z_2, z_3, z_4 = encode_images(np.array([img_1, img_2, img_3, img_4]), VAE.encoder)

      z_mullion = z_2 - z_1

      z_round_mullion = z_3 + z_mullion

      z_square_colorful = (z_1 + z_4)/2

      z_vectors = np.array([ z_1, z_2, z_3, z_mullion, z_round_mullion, z_4, z_square_colorful ])

    # Decode vectors and get image
    #img = visualizer.getLatentSpaceGrid(decoder, z_vectors, img_size = img_size)

    """ RECONSTRUCTION OF REAL DATA """
    img = get_reconstructed_train_data(VAE.combined, PATH_TO_DATASET, N_SAMPLES_PER_IMG, img_size=img_size)

    # Save the image
    visualizer.saveImg(img, PATHS.PATH_TO_OUT_DIRECTORY, 'latent_space_exploration-{}.jpg'.format(i))