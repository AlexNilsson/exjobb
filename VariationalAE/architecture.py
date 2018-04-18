from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model

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

_x = None
_y = None
_z = None

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