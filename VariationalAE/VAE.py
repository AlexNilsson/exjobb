

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os

pixels_amount = 128
batches_size= 8
input_layer_size = pixels_amount*pixels_amount
hidden_1_size =512
n_z = 2
n_epoch = 10
img_rows, img_cols = pixels_amount, pixels_amount

#defining and reshaping data
train_images_path = os.path.join(os.path.dirname(__file__), '../data/shapes/shape')
#print(train_images_path)
train_images_resized_path = os.path.join(os.path.dirname(__file__), '../data/shapes/shape_resized')


#resizing and grayscale
listing = os.listdir(train_images_path)
num_samples = np.size(listing)

for file in listing:
    im = Image.open(train_images_path + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(train_images_resized_path + '\\' + file ) #+ ".jpg"

imlist = os.listdir(train_images_resized_path)
im1= np.array(Image.open(train_images_resized_path + '\\' + imlist[0])) #open one image to get size
m,n = im1.shape[0:2] #size of image
imnbr = len(imlist) #number of images
#matrix to store flattened images
immatrix = np.array([np.array(Image.open(train_images_resized_path + '\\' + im2)).flatten()
    for im2 in imlist], 'f')

print(immatrix.shape)
#--------------------------------------------------------------------
#----------------------------------------------
""" (x_train, y_train), (x_test, y_test) = mnist.load_data() """
x_train = immatrix
x_test = immatrix
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


""" y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255. """

#if noisy images is wanted
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

#encoder
inputs = Input(shape=(input_layer_size,))
h_q =Dense(hidden_1_size, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

#reparametrization trick 
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(batches_size, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps
#sample z Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

#decoder -- P(X|z)
decoder_hidden = Dense(hidden_1_size, activation='relu')
dropout_layer = Dropout(0.2, input_shape=(hidden_1_size,))
decoder_out = Dense(input_layer_size, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

#overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

#encoder model, to encode input into latent variable
encoder = Model(inputs, mu)

#decoder (generator) model, to generate new data given variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
drop = dropout_layer(d_h )
d_out = decoder_out(drop)
decoder = Model(d_in, d_out)

#loss definition
""" def vae_loss(y_true, y_pred):
    calculate loss = reconstruction loss + KL loss for each data in minibatch
    #E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    #D_KL(Q(z|X)) || P(z|X)
    kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) -1. - log_sigma, axis=1)

    return recon + kl """

def vae_loss(inputs, outputs):
    xent_loss = input_layer_size*binary_crossentropy(inputs, outputs)
    kl_loss = - 0.5 * K.mean(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
    return xent_loss + kl_loss

#vae.compile(optimizer='rmsprop', loss=vae_loss)

#models summary
print('vae')
vae.summary()
print('encoder')
encoder.summary()
print('decoder')
decoder.summary()

#training
vae.compile(optimizer='adam', loss=vae_loss)
#vae.fit(x_train, x_train, batch_size=batches_size, nb_epoch=n_epoch)

vae.fit(x_train_noisy, x_train,
        shuffle=True,
        epochs=n_epoch,
        batch_size=batches_size,
        validation_data=(x_test, x_test))

""" vae.fit_generator(train_batches, epochs=n_epoch)
 """
""" #saving model and weights
vae.save('vae_model.h5')
vae.save_weights('vae_weights.h5')
decoder.save_weights('decoder_weights.h5')
encoder.save_weights('encoder_weights.h5') """

""" #plot of class clustering
x_test_encoded = encoder.predict(x_test, batch_size=m)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1]) #, c=y_test
plt.colorbar()
x_test_encoded = encoder.predict_generator(train_batches) """
# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = pixels_amount
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)

plt.show()