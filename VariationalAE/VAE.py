

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os

from callbacks import PlotLatentSpaceProgress
from processing import preProcessImages

""" user preferences """

DATASET = 'windows'
load_saved_weights = False
pixels_amount = 200 #must be dividable by 8
batches_size= 106 #the trainingset must be dividable with batches_size
n_epoch = 10000
hidden_1_size =50
hidden_2_size = 1000 #the flat dense layers before and after z
dropout_amount = 0.4
z_layer_size = 2
noise_factor = 0
save_model_when_done = False

PATH_TO_THIS_DIR = os.path.dirname(__file__)
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, DATASET)
PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, 'saved_weights')

# data paths
train_images_path = os.path.join(PATH_TO_DATASET, '')
train_images_resized_path = os.path.join(PATH_TO_DATASET, 'resized')

# pre process data
preProcessImages(train_images_path, train_images_resized_path,
  convert_to_grayscale=True,
  resize_to=(pixels_amount, pixels_amount)
)

# matrix of flattened images
image_matrix = np.array([
  np.array(
    Image.open(
      os.path.join(train_images_resized_path, img)
    )
  ).flatten() for img in os.listdir(train_images_resized_path)
], 'f')

x_train = image_matrix.astype('float32') / 255.
# adapt this if using `channels_first` image data format
a = [img for img in os.listdir(train_images_resized_path)]
print(len(a))
print(x_train.shape)
print(len(x_train))
x_train = np.reshape(x_train, (len(x_train), pixels_amount, pixels_amount, 1))

x_test = x_train

x_train_ref = x_train
x_test_ref = x_test

print(x_train.shape)
print(x_test.shape)

#if noisy images are wanted
if noise_factor > 0:
  x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
  x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

  x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
  x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
  x_train = x_train_noisy
  x_test = x_test_noisy

"""---------------------------------"""

# returns: ( mu(input_tensor), log_sigma(input_tensor) )
def buildEncoder(input_tensor):
  x = input_tensor

  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  convoluted = x

  """ Variables used for decoders layers sizes """
  convShape = convoluted.shape
  _x = int(convShape[1])
  _y = int(convShape[2])
  _z = int(convShape[3])

  x = Flatten()(x)
  x = Dropout(dropout_amount, input_shape=(hidden_1_size,))(x)
  x = Dense(hidden_1_size, activation='relu')(x)
  x = Dense(hidden_2_size, activation='relu')(x)

  mu = Dense(z_layer_size, activation='linear')(x)
  log_sigma = Dense(z_layer_size, activation='linear')(x)

  return (mu, log_sigma)


def sample_z(args):
  mu, log_sigma = args
  # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
  eps = K.random_normal(shape=(K.shape(mu)[0], z_layer_size), mean=0, stddev=1)
  # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
  return mu + K.exp(log_sigma) * eps


def buildDecoder(input_tensor):
  x = input_tensor

  _x = int(pixels_amount/8)
  _y = int(pixels_amount/8)
  _z = int(8)

  x = Dense(hidden_2_size, activation='relu')(x)
  x = Dense(hidden_1_size, activation='relu')(x)
  x = Dropout(dropout_amount, input_shape=(hidden_1_size,))(x)
  x = Dense(_x*_y*_z, activation='relu')(x)
  x = Reshape((_x,_y,_z))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

  decoder = x
  return decoder

#encoder model, to encode input into latent variable
""" Structure """
inputs = Input(shape=(pixels_amount, pixels_amount, 1))

mu, log_sigma = buildEncoder(inputs)
encoder = Model(inputs, mu)

print('encoder summary')
encoder.summary()

#VAE model, for reconstruction and training
z = Lambda(sample_z)([mu, log_sigma])
outputs = buildDecoder(z)
vae = Model(inputs, outputs)

# Decoder model, for latent space sampling
z_in = Input(shape=(z_layer_size,))
decode_output = buildDecoder(z_in)
decoder = Model(z_in, decode_output)

print('decoder summary')
decoder.summary()

"""---------------------------------"""

# load saved weights
if load_saved_weights:
  before_weight_load = vae.get_weights()
  vae.load_weights(os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), by_name=False)
  after_weight_load = vae.get_weights()
  print('before_weight_load')
  print(before_weight_load)
  print('after_weight_load')
  print(after_weight_load)

#loss definition
""" Flatten input and output layer to use in loss function """
inputs_flat = Flatten()(inputs)
outputs_flat = Flatten()(outputs)

""" loss function """
def vae_loss(inputs_flat, outputs_flat):
	# compute the average MSE error, then scale it up, ie. simply sum on all axes
	reconstruction_loss = K.sum(K.square(outputs_flat-inputs_flat))
	# compute the KL loss
	kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.square(K.exp(log_sigma)), axis=-1)
	# return the average loss over all images in batch
	total_loss = K.mean(reconstruction_loss + kl_loss*0.5)
	return total_loss

""" def vae_loss(inputs_flat, outputs_flat):
	# compute the average MSE error, then scale it up, ie. simply sum on all axes
	reconstruction_loss = K.sum(K.binary_crossentropy(outputs_flat, inputs_flat))
	# compute the KL loss
	kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
	# return the average loss over all images in batch
	total_loss = K.mean(reconstruction_loss + kl_loss)
	return total_loss """

print('vae loss: {}'.format(vae_loss))

#models summary
print('vae')
vae.summary()
print('encoder')
encoder.summary()
print('decoder')
decoder.summary()

#training
vae.compile(optimizer='adam', loss=vae_loss)

latent_space_progress = PlotLatentSpaceProgress(
  model = decoder,
  tiling = 20,
  tile_size = pixels_amount,
  zoom = 1,
  show_plot = False,
  save_plot = True,
  path_to_save_directory = os.path.join(PATH_TO_THIS_DIR, 'epoch_plots'),
  save_name = 'h1_size{}-h2_size{}-dropout{}'.format(hidden_1_size, hidden_2_size, dropout_amount)
  )

weights_checkpoint_callback = ModelCheckpoint(filepath = os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), verbose=1, save_best_only=True, save_weights_only=True)

vae.fit(x_train, x_train_ref,
        shuffle=True,
        epochs=n_epoch,
        batch_size=batches_size,
        validation_data=(x_test, x_test_ref),
        callbacks=[latent_space_progress])
        #callbacks=[latent_space_progress, weights_checkpoint_callback])

if save_model_when_done:
  vae.save(os.path.join(PATH_TO_THIS_DIR, 'vae_model.h5'))
  vae.save_weights(os.path.join(PATH_TO_THIS_DIR,'vae_weights.h5'))
  decoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'decoder_weights.h5'))
  encoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'encoder_weights.h5'))
  decoder.save(os.path.join(PATH_TO_THIS_DIR,'decoder_model.h5'))
