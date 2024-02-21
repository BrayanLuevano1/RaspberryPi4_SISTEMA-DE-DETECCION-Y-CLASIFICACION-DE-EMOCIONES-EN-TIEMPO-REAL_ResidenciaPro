import numpy as np
from PIL import Image
import imageio

def preprocess_input(x, v2=True):
    x = 2.0 * (x.astype('float32') / 255.0 - 0.5) if v2 else x.astype('float32') / 255.0
    return x

def _imread(image_name):
    return np.array(Image.open(image_name))

def _imresize(image_array, size):
    return np.array(Image.fromarray(image_array).resize(size))

def to_categorical(integer_classes, num_classes=2):
    return np.eye(num_classes)[integer_classes.astype('int')]
