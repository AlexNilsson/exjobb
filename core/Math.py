import numpy as np

def angle_between_vectors(a, b):
  """ Smallest angle between two n-dimensional Vectors """
  cos_angle = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
  return np.arccos(np.clip(cos_angle, -1, 1))

def vectorLerp(a, b, fraction):
  """ Linear Iterpolation between two Vectors """
  return a + fraction * (b - a)

def vectorSlerp(a, b, fraction):
  """ Spherical Interpolation between two Vectors"""
  theta = angle_between_vectors(a, b)
  return a * np.sin((1-fraction) * theta) / np.sin(theta) + b * np.sin(fraction * theta) / np.sin(theta)
