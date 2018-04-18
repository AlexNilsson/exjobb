

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

DATASET = 'shapes'
load_saved_weights = False
pixels_amount = 40 #must be dividable by 8
batches_size= 100 #the trainingset must be dividable with batches_size
n_epoch = 100
hidden_1_size =200
hidden_2_size = 100 #the flat dense layers before and after z
dropout_amount = 0.4
z_layer_size = 2
noise_factor = 0
save_model_when_done = False


PATH_TO_THIS_DIR = os.path.dirname(__file__)
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, DATASET)
PATH_TO_SAVED_WEIGHTS = os.path.join(PATH_TO_THIS_DIR, 'saved_weights')

# data paths
train_images_path = os.path.join(PATH_TO_DATASET, 'shape')
train_images_resized_path = os.path.join(PATH_TO_DATASET, 'shape_resized')

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
x_train = np.reshape(x_train, (len(x_train), pixels_amount, pixels_amount, 1))

x_test = x_train

x_train_ref = x_train
x_test_ref = x_test

print(x_train.shape)
print(x_test.shape)

#if noisy images is wanted

if noise_factor > 0:
  x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
  x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

  x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
  x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
  x_train = x_train_noisy
  x_test = x_test_noisy

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------

#encoder model, to encode input into latent variable
""" Convolution layers """
conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
pooling_1 = MaxPooling2D((2, 2), padding='same')
conv_2 = Conv2D(8, (3, 3), activation='relu', padding='same')
pooling_2 = MaxPooling2D((2, 2), padding='same')
conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')
pooling_3 = MaxPooling2D((2, 2), padding='same')

""" Flat layers """
flat_hidden_1 =Dense(hidden_1_size, activation='relu')
flat_hidden_2 = Dense(hidden_2_size, activation='relu')
flat_2 = Dense(z_layer_size, activation='linear')
flat_3 = Dense(z_layer_size, activation='linear')
dropout_layer_encoder = Dropout(dropout_amount, input_shape=(hidden_1_size,)) #can be placed between the flat dense layers if needed
""" Structure """
inputs = Input(shape=(pixels_amount, pixels_amount, 1))
x = conv_1(inputs)
x = pooling_1(x)
x = conv_2(x)
x = pooling_2(x)
x = conv_3(x)
convoluted = pooling_3(x)

convoluted_flat = Flatten()(convoluted)
drop_encoder = dropout_layer_encoder(convoluted_flat)
h_q =flat_hidden_1(drop_encoder)
h_q_2 = flat_hidden_2(h_q)
mu = flat_2(h_q_2)
log_sigma = flat_3(h_q_2)
encoder = Model(inputs, mu)

print('encoder summary')
encoder.summary()

""" Variables used for decoders layers sizes """
convShape = convoluted.shape
_x = int(convShape[1])
_z = int(convShape[3])



def sample_z(args):
  mu, log_sigma = args
  print(log_sigma)
  # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
  eps = K.random_normal(shape=(K.shape(mu)[0], z_layer_size), mean=0, stddev=1)
  # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
  return mu + K.exp(log_sigma) * eps

#sample z Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])


#decoder -- P(X|z) (generator) model, to generate new data given variable z
""" Layers """
decoder_hidden_2 = Dense(hidden_2_size, activation='relu')
dropout_layer_decoder = Dropout(dropout_amount, input_shape=(hidden_2_size,)) #can be placed between the flat dense layers if needed
decoder_hidden = Dense(hidden_1_size, activation='relu')
d_h = Dense(_x*_x*_z, activation='relu')
d_h_reshaped = Reshape((_x,_x,_z))
decon_1 = Conv2D(8, (3, 3), activation='relu', padding='same')
upsamp_1 = UpSampling2D((2, 2))
decon_2 = Conv2D(8, (3, 3), activation='relu', padding='same')
upsamp_2 = UpSampling2D((2, 2))
decon_3 = Conv2D(16, (3, 3), activation='relu', padding='same')
upsamp_3 = UpSampling2D((2, 2))
decon_4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

""" Structure """
d_in = Input(shape=(z_layer_size,))
x = decoder_hidden_2(d_in)
x = dropout_layer_decoder(x)
x = decoder_hidden(x)
x = d_h(x)
x = d_h_reshaped(x)
x = decon_1(x)
x = upsamp_1(x)
x = decon_2(x)
x = upsamp_2(x)
x = decon_3(x)
x = upsamp_3(x)
decode_output = decon_4(x)
decoder = Model(d_in, decode_output)
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------

#VAE model, for reconstruction and training
""" Structure """
x = decoder_hidden_2(z)
x = decoder_hidden(x)
x = dropout_layer_decoder(x)
x = d_h(x)
x = d_h_reshaped(x)
x = decon_1(x)
x = upsamp_1(x)
x = decon_2(x)
x = upsamp_2(x)
x = decon_3(x)
x = upsamp_3(x)
outputs = decon_4(x)
vae = Model(inputs, outputs)

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
# load saved weights
if load_saved_weights:
  before_weight_load = vae.get_weights()
  vae.load_weights(os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), by_name=False)
  after_weight_load = vae.get_weights()
  print('before_weight_load')
  print(before_weight_load)
  print('after_weight_load')
  print(after_weight_load)
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------

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
  show_plot = True,
  save_plot = True,
  path_to_save_directory = os.path.join(PATH_TO_THIS_DIR, 'epoch_plots'),
  save_name = 'h1_size{}-h2_size{}-dropout{}.jpg'.format(hidden_1_size, hidden_2_size, dropout_amount)
  )

weights_checkpoint_callback = ModelCheckpoint(filepath = os.path.join(PATH_TO_SAVED_WEIGHTS, 'weight.hdf5'), verbose=1, save_best_only=True, save_weights_only=True)

vae.fit(x_train, x_train_ref,
        shuffle=True,
        epochs=n_epoch,
        batch_size=batches_size,
        validation_data=(x_test, x_test_ref),
        callbacks=[latent_space_progress, weights_checkpoint_callback])

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
if save_model_when_done:
  vae.save(os.path.join(PATH_TO_THIS_DIR, 'vae_model.h5'))
  vae.save_weights(os.path.join(PATH_TO_THIS_DIR,'vae_weights.h5'))
  decoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'decoder_weights.h5'))
  encoder.save_weights(os.path.join(PATH_TO_THIS_DIR,'encoder_weights.h5'))
  decoder.save(os.path.join(PATH_TO_THIS_DIR,'decoder_model.h5'))
