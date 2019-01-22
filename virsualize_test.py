import os
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

file_path = '/home/soya/Desktop/pred_test_abnormal.npy'
file = np.load(file_path)
file = np.squeeze(file, axis=0)

image_mean = np.load('/media/soya/Newsmy/datasets/UCSDped/UCSDped1/training_frames/image_mean_227.npy')
normal_savepath = '/home/soya/Desktop/pred_test/'
abnormal_savepath = '/home/soya/Desktop/pred_test_abnormal/'

def slice_image(file, savepath):
    for num, image in enumerate(file):
        np.save(os.path.join(savepath, 'test_image{:02d}.npy'.format(num+1)), image)
        print('test_image{:02d} saved'.format(num+1))

abnormal_file_path = '/home/soya/Desktop/pred_test_abnormal'
normal_file_path = '/home/soya/Desktop/pred_test'

def visualize_volume_images(filepath):
    for image in sorted(os.listdir(filepath)):
        image = os.path.join(filepath, image)
        image_value = np.load(image)
        scaler = preprocessing.StandardScaler()
        scaler.fit(image_value)
        test_image = scaler.inverse_transform(image_value)
        #test_image = test_image + image_mean
        test_image = test_image * 255

        plt.imshow(test_image)
        plt.show()

if __name__ == '__main__':
    visualize_volume_images(normal_file_path)