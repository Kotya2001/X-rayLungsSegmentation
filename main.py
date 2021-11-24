import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def image_resize(path, size):
    return cv2.resize(path, size)


def get_segmentation(model, dir):
    number_of_images = len(dir)
    for number in range(number_of_images):
        image = np.array(np.stack((cv2.resize(cv2.imread(dir[number], 0),
                                              (256, 256)),), axis=-1))
        image = np.expand_dims(image, axis=0)

        image = image / 255.

        prediction = model.predict(image)

        yield prediction[0, ..., 0]


add = '/Users/ilabelozerov/Downloads/archive-2/data/Lung Segmentation/test'
dir = [i for i in os.listdir(add)]
paths = ['/Users/ilabelozerov/Downloads/archive-2/data/Lung Segmentation/test/' + i for i in dir]
size = (256, 256)
model = load_model('../Unet')

plt.imshow(next(get_segmentation(model, paths)))
plt.show()
