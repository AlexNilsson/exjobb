

from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import pandas as pd
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
""" user preferences """
pixels_amount = 28 #must be dividable by 8
batchsize = 200 #the trainingset must be dividable with batches_size
n_epoch = 10000
hidden_1_size =100 
hidden_2_size = 50 #the flat dense layers before and after z
dropout_amount = 0.3
z_layer_size = 2

""" adjust discriminator layersize to match complexity of the decoder """
disc_1_size = 50
disc_2_size = 100
disc_3_size = 100
disc_4_size = 50
disc_5_size = 50

input_layer_size = pixels_amount*pixels_amount
img_rows, img_cols = pixels_amount, pixels_amount #what size images are reshaped to
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------


train = pd.read_csv("./data/csv_mnist_gan/train.csv").values

x_train = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
x_train = x_train.astype(float)
x_train /= 255.0

x_train = x_train[:2000]

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#defining and reshaping data
""" train_images_path = os.path.join(os.path.dirname(__file__), '../data/shapes/shape')
train_images_resized_path = os.path.join(os.path.dirname(__file__), '../data/shapes/shape_resized')
train_images_resized_bigger_path = os.path.join(os.path.dirname(__file__), '../data/shapes/shape_resized_bigger')
#resizing and grayscale of training images 
listing = os.listdir(train_images_path)
num_samples = np.size(listing)

for file in listing:
    im = Image.open(train_images_path + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(train_images_resized_path + '\\' + file ) #+ ".jpg"
    img_re = im.resize((img_rows*2,img_cols*2))
    gray_re = img_re.convert('L')
    gray_re.save(train_images_resized_bigger_path + '\\' + file )

imlist = os.listdir(train_images_resized_path)
im1= np.array(Image.open(train_images_resized_path + '\\' + imlist[0])) #open one image to get size
m,n = im1.shape[0:2] #size of image
imnbr = len(imlist) #number of images
#matrix to store flattened images
immatrix = np.array([np.array(Image.open(train_images_resized_path + '\\' + im2)).flatten()
    for im2 in imlist], 'f') """

""" imlist = os.listdir(train_images_resized_bigger_path)
im1= np.array(Image.open(train_images_resized_bigger_path + '\\' + imlist[0])) #open one image to get size
m,n = im1.shape[0:2] #size of image
imnbr = len(imlist) 
immatrix_bigger = np.array([np.array(Image.open(train_images_resized_bigger_path + '\\' + im2)).flatten()
    for im2 in imlist], 'f')
print(immatrix.shape)
print(immatrix_bigger.shape) """

""" x_test = immatrix_bigger """
""" x_train = immatrix
x_test = immatrix
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), pixels_amount, pixels_amount, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), pixels_amount, pixels_amount, 1))  # adapt this if using `channels_first` image data format
print(x_train.shape)
print(x_test.shape) """

#if noisy images is wanted 
""" 
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0) """
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


#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#reparametrization trick to sample z 
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], z_layer_size), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps

""" def sample_z(mu, log_sigma):
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    eps = K.random_normal(shape=(K.shape(mu)[0], z_layer_size), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mu + K.exp(log_sigma) * eps """

#sample z Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------


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
decon_3 = Conv2D(16, (3, 3), activation='relu')#, padding='same'
upsamp_3 = UpSampling2D((2, 2))
decon_4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

""" upsamp_4 = UpSampling2D((2, 2))
decon_5 = Conv2D(1, (3, 3), activation='sigmoid', padding='same') """

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
""" x = decon_4(x)
x =upsamp_4(x) """
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
""" x = decon_4(x)
x = upsamp_4(x) """
outputs = decon_4(x)
vae = Model(inputs, outputs)


#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#to load weights

""" before_weight_load = vae.get_weights()
vae.load_weights('./VariationalAE/saved_weights/weight.hdf5', by_name=False)
after_weight_load = vae.get_weights()
print('before_weight_load')
print(before_weight_load)
print('after_weight_load')
print(after_weight_load) """
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------

#loss definition
""" Flatten input and output layer to use in loss function """
inputs_flat = Flatten()(inputs)
outputs_flat = Flatten()(outputs)

""" VAE loss function """
vae_loss = reconstruction_loss = K.sum(K.square(outputs_flat-inputs_flat))


print('vae loss')
print(vae_loss)
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#discriminator and discriminator encoder
#layers
disc_1 = Dense(disc_1_size, activation='relu')
disc_2 = Dense(disc_2_size, activation='relu')
disc_3 = Dense(disc_3_size, activation='relu')
disc_4 = Dense(disc_4_size, activation='relu')
disc_5 = Dense(disc_5_size, activation='relu')
disc_out = Dense(1, activation='sigmoid')

#structure discriminator
discriminator_input = Input(shape=(z_layer_size,))
x = disc_1(discriminator_input)
x = disc_2(x)
x = disc_3(x)
x = disc_4(x)
#x = disc_5(x)
discriminator_output = disc_out(x)
discriminator = Model(discriminator_input, discriminator_output)

#structure discriminatorencoder
x = disc_1(z)
x = disc_2(x)
x = disc_3(x)
x = disc_4(x)
#x = disc_5(x)
discriminator_e_output = disc_out(x)
discriminator_encoder = Model(inputs, discriminator_e_output)
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------

#models summary
print('vae')
vae.summary()
print('encoder')
encoder.summary()
print('decoder')
decoder.summary()
print('discriminator')
discriminator.summary()
print('discriminator encoder')
discriminator_encoder.summary()

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#training
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator_encoder.compile(optimizer='adam', loss='binary_crossentropy')
vae.compile(optimizer='adam', loss='binary_crossentropy')
encoder.compile(optimizer='adam', loss='binary_crossentropy')
decoder.compile(optimizer='adam', loss='binary_crossentropy')

#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#Callback functions to plot and save images and weights
class z_sampling_progress(Callback):
 def on_epoch_begin(self, epoch, logs):
    n = 20  # figure with nxn images
    visualisation_size = pixels_amount
    figure = np.zeros((visualisation_size * n, visualisation_size * n))
    # sample n points within [linspace range] standard deviations
    grid_x = np.linspace(-1, 1, n)
    grid_y = np.linspace(-1, 1, n)
    

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            visualisation_image = x_decoded[0].reshape(visualisation_size, visualisation_size)
            figure[i * visualisation_size: (i + 1) * visualisation_size,
                j * visualisation_size: (j + 1) * visualisation_size] = visualisation_image
    c.imshow('./VariationalAE/epoch_plots/tmp', figure)
    c.waitKey(1)
    figure_to_file = figure*255 #reshape to get the right range when saving image to file
    figure_to_file = figure_to_file.astype('uint8')
    name_of_image = './VariationalAE/epoch_plots/{}.jpg'.format(epoch)
    c.imwrite(name_of_image, figure_to_file)
    def on_epoch_end(self, epoch, logs):
        c.destroyAllWindows()

sampling_progress_callback = z_sampling_progress()
weights_checkpoint_callback = ModelCheckpoint(filepath='./VariationalAE/saved_weights/weight.hdf5', verbose=1, save_best_only=True, save_weights_only=True)

#weights_checkpoint = ModelCheckpoint(filepath='./VariationalAE/saved_weights/weight-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)

""" vae.fit(x_train, x_test,
        shuffle=True,
        epochs=n_epoch,
        batch_size=batches_size,
        validation_data=(x_train, x_test),
        callbacks=[sampling_progress_callback, weights_checkpoint_callback]) """

def imagegrid(dec, epochnumber):        
        fig = plt.figure(figsize=[20, 20])
        
        for i in range(-5, 5):
            for j in range(-5,5):
                topred = np.array((i*0.5,j*0.5))
                topred = topred.reshape((1, 2))
                img = dec.predict(topred)
                img = img.reshape((28, 28))
                ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        
        fig.savefig('./VariationalAE/epoch_plots/{}.jpg'.format(epochnumber))
        """ plt.show()
        plt.close(fig) """

def epoch_vis(self, epoch):
    n = 20  # figure with nxn images
    visualisation_size = pixels_amount
    figure = np.zeros((visualisation_size * n, visualisation_size * n))
    # sample n points within [linspace range] standard deviations
    grid_x = np.linspace(-1, 1, n)
    grid_y = np.linspace(-1, 1, n)
    

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            visualisation_image = x_decoded[0].reshape(visualisation_size, visualisation_size)
            figure[i * visualisation_size: (i + 1) * visualisation_size,
                j * visualisation_size: (j + 1) * visualisation_size] = visualisation_image
    """ c.imshow('./VariationalAE/epoch_plots/tmp', figure)
    c.waitKey(1) """
    figure_to_file = figure*255 #reshape to get the right range when saving image to file
    figure_to_file = figure_to_file.astype('uint8')
    name_of_image = './VariationalAE/epoch_plots/{}.jpg'.format(epoch)
    c.imwrite(name_of_image, figure_to_file)

def settrainable(model, toset):
    for layer in model.layers:
        layer.trainable = toset
    model.trainable = toset

print('x train shape')
print(x_train.shape)
for epochnumber in range(n_epoch):
    np.random.shuffle(x_train)
    
    for i in range(int(len(x_train) / batchsize)):
        """ training stage 1 -- reconstruction of image """
        settrainable(vae, True)
        settrainable(encoder, True)
        settrainable(decoder, True)
        
        batch = x_train[i*batchsize:i*batchsize+batchsize]
        vae.train_on_batch(batch, batch) #trains the autoencoder to reconstruct imput images
        
        """ training stage 2 -- differ between wanted and unwanted distribution """
        settrainable(discriminator, True)
        batchpred = encoder.predict(batch) #the encoder predicts the distribution of z-values for batch
        fakepred = np.random.standard_normal((batchsize,2)) #the wanted distibution
        discbatch_x = np.concatenate([batchpred, fakepred]) #concats the predicted and wanted distribution
        discbatch_y = np.concatenate([np.zeros(batchsize), np.ones(batchsize)])
        discriminator.train_on_batch(discbatch_x, discbatch_y) #trains the discriminator to differ between wanted and unwanted (from encoder z) distribution
        
        """ training stage 3 -- encode z-values to fit in the wanted distribution """
        settrainable(discriminator_encoder, True)
        settrainable(encoder, True)
        settrainable(discriminator, False) #using the weights from previous training stage (we just want to update the encoders weights)
        discriminator_encoder.train_on_batch(batch, np.ones(batchsize)) #trains the encoder to code z-values to fit in the wanted distribution
    
    print(epochnumber, 'of' '{}'.format(n_epoch))
    print("Reconstruction Loss:", vae.evaluate(x_train, x_train, verbose=1))
    print("Adverserial Loss:", discriminator_encoder.evaluate(x_train, np.ones(len(x_train)), verbose=1))

    if epochnumber % 10 == 0:
        #imagegrid(decoder, epochnumber)
        epoch_vis(decoder, epochnumber)
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
#saving model and weights

""" vae.save('vae_model.h5')
decoder.save('decoder_model.h5') """

vae.save_weights('vae_weights.h5')
discriminator.save_weights('discriminator_weight.h5')
decoder.save_weights('decoder_weights.h5')
encoder.save_weights('encoder_weights.h5')
discriminator_encoder.save_weights('discriminator_encoder_weight.h5')
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------
# visualisation
""" n = 20  # figure with 15x15 images
visualisation_size = pixels_amount
figure = np.zeros((visualisation_size * n, visualisation_size * n))
# we will sample n points within [linspace range] standard deviations
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        visualisation_image = x_decoded[0].reshape(visualisation_size, visualisation_size)
        figure[i * visualisation_size: (i + 1) * visualisation_size,
               j * visualisation_size: (j + 1) * visualisation_size] = visualisation_image

print(visualisation_image)
print(visualisation_image.shape)
print(figure)
plt.figure(figsize=(10, 10))
plt.imshow(figure)

plt.savefig(
  "./VariationalAE/progress_images/samples{}-epoch{}-h1_size{}-h2_size{}-dropout{}.jpg".format(n, n_epoch, hidden_1_size, hidden_2_size, dropout_amount)
  )
plt.show() """


#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------