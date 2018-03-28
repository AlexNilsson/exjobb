# Packages
import os
import numpy as np
import matplotlib.pyplot as plt

# Keras
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *

# Project Modules
from modules import data
from modules import visualization

model_file_path = os.path.join(os.path.dirname(__file__), 'models/medical_trial_mode.h5')

# Data
train_path = os.path.join(os.path.dirname(__file__), '../data/cat_or_dog/train')
valid_path = os.path.join(os.path.dirname(__file__), '../data/cat_or_dog/valid')
test_path = os.path.join(os.path.dirname(__file__), '../data/cat_or_dog/test')

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['dog','cat'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['dog','cat'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['dog','cat'], batch_size=10)
#test_batches.class_indices

# fine tuining of existing model
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

# Convert model to sequential
model = Sequential()
for layer in vgg16_model.layers:
  model.add(layer)

# Customize last layer

model.layers.pop()

# Locking existing layers; These should not be trained; treat as constant
for layer in model.layers:
  layer.trainable = False

# Add last custom layer on top
model.add(Dense(2, activation='softmax'))

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

# Plot prediction
test_imgs, test_labels = next(test_batches)
visualization.plotImages(test_imgs, test_labels)

# predict
predictions = model.predict_generator(test_batches, steps=1, verbose=0)

visualization.plotConfusionMatrix(test_labels[:,0], predictions[:,0])

plt.show()




def demo2():
  # Plot train data
  imgs, labels = next(train_batches)
  visualization.plotImages(imgs, labels)

  # model
  model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(2, activation='softmax')
  ])

  model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)



  # Plot prediction
  test_imgs, test_labels = next(test_batches)
  visualization.plotImages(test_imgs, test_labels)

  # predict
  predictions = model.predict_generator(test_batches, steps=1, verbose=0)

  visualization.plotConfusionMatrix(test_labels[:,0], predictions[:,0])

  plt.show()

def train():

  # Create Model
  model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax'),
  ])

  model.summary()

  model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Train model
  (train_data, train_labels) = data.getProcessedTrainData()

  model.fit(train_data, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

  # Save model
  model.save(model_file_path)

def demo():
  # Load Model
  from keras.models import load_model
  model = load_model(model_file_path)

  model.summary()

  # Only weights
  #model.sample_weights('models/my_model_weights.h5')
  #model2.load_weights('models/my_model_weights.h5')

  # Predict
  (test_data, test_labels) = data.getProcessedTestData()

  predictions = model.predict(test_data, batch_size=10, verbose=0)

  rounded_predictions = model.predict_classes(test_data, batch_size=10, verbose=0)

  # PLOT CONFUSION MATRIX

  cm_plot_labels = ['no_side_effect', 'had_side_effect']

  plt.figure()

  visualization.plotConfusionMatrix(test_labels, rounded_predictions, plot_labels=cm_plot_labels, plot_title='Confusion Matrix')

  plt.show()
