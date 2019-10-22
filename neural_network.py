import sys, os, inspect

import numpy as np

from scipy.io import loadmat
from scipy import optimize

import pandas as pd

import idx2numpy

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.image import NonUniformImage
from matplotlib import cm

plt.style.use('ggplot')


# SET UP LOGGER ===============================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(asctime)s: %(filename)s: %(lineno)s:\n%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
def load_idx_file(
        file_path: str,
        ):
    with open(file_path, 'rb') as f_in:
        array = idx2numpy.convert_from_file(f_in)
    return array


def import_mnist_data(
        mnist_data_path: str
        ):
    mnist_data = ()
    data_file_list = os.listdir(mnist_data_path)
    for f in data_file_list:
        file_path = os.path.join(mnist_data_path, f)
        mnist_data += (load_idx_file(file_path),)
    X_train, y_train, X_test, y_test = mnist_data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    beta = np.zeros(X_train.shape[1])
    logger.info(
        f"\tX train {X_train.shape}\n"
        f"\ty train {y_train.shape}\n"
        f"\tX test {X_test.shape}\n"
        f"\ty test {y_test.shape}\n"
        f"\tbeta {beta.shape}\n")
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        beta
        )


np.random.seed(0)
def get_single_digit_data(image):
    '''
    converts image_vector (784,) into a matrix(28, 28)
    '''
#     by default order='C', but in this case the lines are assembled in wrong order
    return np.reshape(image, (-1, int(image.shape[0] ** .5)), order='C')

def show_random_digit(
        images: np.ndarray,
        sample_size: int,
        ):
    random_index = np.random.randint(sample_size)
    fig = plt.figure(figsize=(2, 2))
    digit_image = get_single_digit_data(images[random_index])
    plt.imshow(digit_image, cmap = 'gist_gray')
    plt.axis('off')
    return plt.show()

def show_hundred_digits(
        visualisation_set):
    fig = plt.figure(figsize=(6, 6))
    for row in range(10):
        for column in range(10):
            digit = get_single_digit_data(
                visualisation_set[10 * row + column])
            sub = plt.subplot(10, 10, 10 * row + column + 1)
            sub.axis('off')
            sub.imshow(
                digit,
                cmap='gist_gray')
    return plt.show()