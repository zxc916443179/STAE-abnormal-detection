import os
import skvideo.io
import skvideo.datasets
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import cv2

video_root_path = '/home/soya/datasets'
data_path = os.path.dirname(os.path.abspath(__file__))
size = (227, 227)

def video_to_frame(dataset,train_or_test):
    video_path = os.path.join(video_root_path, dataset, '{}_videos'.format(train_or_test))
    frame_path = os.path.join(data_path, dataset, '{}_frames'.format(train_or_test))
    os.makedirs(frame_path, exist_ok=True)
    for video_file in os.listdir(video_path):
        print(os.path.join(video_path, video_file))
        if video_file.lower().endswith(('.avi', '.mp4')): # loading  video
            vid_frame_path = os.path.join(frame_path, os.path.basename(video_file).split('.')[0])
            os.makedirs(vid_frame_path, exist_ok=True)
            vidcap = skvideo.io.vreader(os.path.join(video_path, video_file)) # loading video
            count = 1
            for image in vidcap: # read frames from video
                image = resize(image, size, mode='reflect')
                imsave(os.path.join(vid_frame_path, '{:05d}.jpg'.format(count)), image)
                count += 1
            print(count)

if __name__ == "__main__":
    video_to_frame('avenue', 'training')
    video_to_frame('avenue', 'testing')