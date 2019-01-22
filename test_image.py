from skimage.io import imread
import numpy as np
import skimage
from skimage import io
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image

image_path = '/home/soya/datasets/UCSDped/UCSDped1/training_frames/Train001/001.tif'
file_path = '/home/soya/datasets/UCSDped/UCSDped1/training_frames/image_mean_227.npy'

image_mean = np.load(file_path)
image_mean = image_mean * 255
image = Image.fromarray(image_mean)
plt.imshow(image)
plt.show()
print(image_mean)
print(image)
