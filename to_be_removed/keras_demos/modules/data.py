import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

def generateTrainingData():
  train_labels = []
  train_samples = []

  for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

  for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

  return (train_samples, train_labels)


def generateTestData():
  test_labels = []
  test_samples = []

  for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

  for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

  return (test_samples, test_labels)

def getProcessedTrainData():
  (train_samples, train_labels) = generateTrainingData()

  # To NumpArray and Reshape
  train_samples = np.array(train_samples)
  train_labels = np.array(train_labels)

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

  return (scaled_train_samples, train_labels)

def getProcessedTestData():
  (test_samples, test_labels) = generateTestData()

  # To NumpArray and Reshape
  test_samples = np.array(test_samples)
  test_labels = np.array(test_labels)

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

  return (scaled_test_samples, test_labels)
