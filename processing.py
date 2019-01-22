import numpy as np
import os
from skimage.io import imread
import h5py
import cv2
from sklearn import preprocessing

data_root_path = '/media/soya/Newsmy/datasets/UCSDped'
test_folder = '/media/soya/Newsmy/datasets/UCSDped/UCSDped1/testing_frames'
test_h5_path = os.path.join(data_root_path, 'h5_data/test')

# load txt as list
def get_data_files(file):
    return [line.rstrip() for line in file.readlines()]

# caculating global mean value
def pixel_mean(dataset, data_root_path):
    frame_path = os.path.join(data_root_path, dataset, 'training_frames')
    image_sum = np.zeros((227, 227)).astype('float64')
    f = open(os.path.join(data_root_path, dataset, 'training_frames/training_list.txt'))
    count = 0

    for frame_folder in get_data_files(f):
        if not frame_folder.startswith('.'): # elinminate hidden folder
            folder_path = os.listdir(os.path.join(frame_path, frame_folder))
            for image in folder_path:
                image_name = os.path.join(frame_path, frame_folder, image) # loading image path
                if os.path.splitext(image_name)[1] == '.tif':
                    image_value = imread(image_name, as_gray=True) # loading image as npy array
                    image_value = cv2.resize(image_value, (227, 227), interpolation=cv2.INTER_CUBIC)
                    image_sum += image_value
                    count += 1

    image_mean = (image_sum/count).astype(np.float64)
    np.save(os.path.join(frame_path, 'image_mean_227.npy'), image_mean)

# subtract global mean value and normalize
def image_subtract(dataset, data_root_path):
    training_frame_path = os.path.join(data_root_path, dataset, 'training_frames')
    image_mean = np.load(os.path.join(training_frame_path, 'image_mean_227.npy'))
    f = open(os.path.join(training_frame_path, 'training_list.txt'))

    for frame_folder in get_data_files(f):
        training_frames_vid = []  # saving processed training frame as .npy
        if os.path.isfile(os.path.join(training_frame_path, 'training_frames_{}.npy'.format(frame_folder))): continue
        if not frame_folder.startswith('.'):
            for image in os.listdir(os.path.join(training_frame_path, frame_folder)):
                if not image.startswith('.'):
                    image_name = os.path.join(training_frame_path, frame_folder, image)
                    image_value = imread(image_name, as_gray=True)
                    image_value = cv2.resize(image_value, (227, 227), interpolation=cv2.INTER_CUBIC).astype('float64')
                    image_value -= image_mean # extract mean
                    preprocessed_image = preprocessing.scale(image_value) # (0,1) normalization
                    training_frames_vid = np.append(training_frames_vid, preprocessed_image) # saving extracted value into list

            training_frames_vid = np.array(training_frames_vid)
            # one .npy for one training folder
            np.save(os.path.join(training_frame_path, 'training_frames_{}.npy'.format(frame_folder)), training_frames_vid)
            print('training_frames_{}.npy successfully saved.'.format(frame_folder))
    # substract test image
    testing_frame_path = os.path.join(data_root_path, dataset, 'testing_frames')
    f = open(os.path.join(test_folder, 'testing_list.txt'), 'r')

    #for frame_folder in os.listdir(testing_frame_path):
    for frame_folder in get_data_files(f):
        testing_frame_vid = []  # saving processed testing frame as .npy
        print(os.path.join(testing_frame_path, frame_folder))
        if os.path.isfile(os.path.join(testing_frame_path, 'testing_frames_{}.npy'.format(frame_folder))): continue
        for image in sorted(os.listdir(os.path.join(testing_frame_path, frame_folder))):
            if not image.startswith('.'):
                image_name = os.path.join(testing_frame_path, frame_folder, image)
                image_value = imread(image_name, as_gray=True)
                image_value = cv2.resize(image_value, (227, 227), interpolation=cv2.INTER_CUBIC).astype('float64')
                image_value -= image_mean
                preprocessed_image = preprocessing.scale(image_value) # (0,1) normalization
                testing_frame_vid = np.append(testing_frame_vid, preprocessed_image)

        testing_frame_vid = np.array(testing_frame_vid)
        np.save(os.path.join(testing_frame_path, 'testing_frames_{}.npy'.format(frame_folder)), testing_frame_vid)
        print('testing_frames_{}.npy successfully saved.'.format(frame_folder))

# bulid frames into volume with different stride
def build_volume(file, stride=1, time_length=10):
    data_frames = np.load(file)
    data_frames = data_frames.reshape(-1, 227, 227)  # 200x227x227
    num_frames = data_frames.shape[0]

    data_only_frames = np.zeros((num_frames-stride*(time_length-1), time_length, 227, 227)).astype('float64')
    i = 0
    vol = 0
    while (i + stride*(time_length-1)+1 <= num_frames):
        temp_data = data_frames[i : i + stride * (time_length - 1) + 1 : stride]
        #print(temp_data.shape)
        data_only_frames[vol] = temp_data
        vol += 1
        i += 1
    return data_only_frames

def build_test_volume(file, time_length=10):
    data_frames = np.load(file)
    data_frames = data_frames.reshape(-1, 227, 227)
    num_frames = data_frames.shape[0]
    num_volumes = num_frames // time_length

    data_only_frames = np.zeros(shape=(num_volumes, time_length, 227, 227)).astype('float64')
    i = 0
    while (i < num_volumes):
        start_idx = i * time_length
        end_idx = (i+1) * time_length
        temp_data = data_frames[start_idx:end_idx, :, :]
        data_only_frames[i] = temp_data
        i += 1
    if (num_frames % time_length != 0):
        #data_only_frames.resize(num_volumes+1, axis=0)
        real_num = num_frames % time_length
        print(end_idx, real_num)
        temp_data = np.zeros((1, time_length, 227, 227))
        temp_data[:, :real_num, :] = data_frames[end_idx:, :, :]
        temp_data[:, real_num:, :] = data_frames[real_num - 1, :]  # broadcast
        data_only_frames = np.concatenate((data_only_frames, temp_data), axis=0)
    return data_only_frames

# build every .avi into volumes and load into small h5 files, grouped by video ID
def build_h5(dataset, data_root_path, training_or_testing):
    frame_path = os.path.join(data_root_path, dataset, '{}_frames'.format(training_or_testing))
    h5_root_path = os.path.join(data_root_path, dataset, '{}_frames_hdf5'.format(training_or_testing))
    if not os.path.isdir(h5_root_path): os.makedirs(h5_root_path, exist_ok=True)
    for file in sorted(os.listdir(frame_path)):
        if (file.endswith('.npy') and file.startswith('{}'.format(training_or_testing))):
            data_name = os.path.join(frame_path, file)
            print(data_name)
            if training_or_testing == 'training':
                # data concatenate
                volume_stride_1 = build_volume(data_name, stride=1) # 191x10x227x227
                volume_stride_2 = build_volume(data_name, stride=2) # 182x10x227x227
                volume_stride_3 = build_volume(data_name, stride=3) # 173x10x227x227
                volume_concatenate = np.concatenate((volume_stride_1, volume_stride_2, volume_stride_3), axis=0) # 546x10x227x227
            else:
               #test_volume = build_test_volume(data_name) # 20x10x227x227
                test_volume = build_volume(data_name, stride=1) # 191x10x227x227

            with h5py.File(os.path.join(h5_root_path, '{0}_{1}.h5'.format(dataset, file.rstrip('.npy')))) as f:
                if training_or_testing == 'training':
                    np.random.shuffle(volume_concatenate) # shuffle training data
                    f['data'] = volume_concatenate
                else:
                    f['data'] = test_volume
                    print('{0}_{1}.h5'.format(dataset, file.rstrip('.npy')))
                    print(f['data'].shape)

# combine small h5 datasets as one, if test, not combine
def combine_dataset(dataset, data_root_path, training_or_testing):
    # path of loading data
    h5_data_path = os.path.join(data_root_path, dataset, '{}_frames_hdf5'.format(training_or_testing))
    h5_outpath = os.path.join(data_root_path, dataset, 'data_h5')
    if not os.path.isdir(h5_outpath): os.makedirs(h5_outpath, exist_ok=True)
    file_combined = h5py.File(os.path.join(h5_outpath, '{0}_{1}.h5'.format(dataset, training_or_testing)), 'w')
    file_list = sorted([os.path.join(h5_data_path, file) for file in os.listdir(h5_data_path)])

    total_rows = 0 # counting total data number

    # loading all small h5 data
    for n, f in enumerate(file_list):
        print(n, f)
        file = h5py.File(f, 'r')
        data = file['data']
        total_rows += data.shape[0]

        # saving small data into combined h5
        if n == 0: # create first dataset
            # initialize dataset creating
            combined_dataset = file_combined.create_dataset('data', shape=(total_rows, 10, 227, 227),
                                                            maxshape=(None, 10, 227, 227))
            combined_dataset[:, :] = data
            where_to_start_appending = total_rows
            #print(where_to_start_appending, total_rows, data.shape)

        else:
            print(where_to_start_appending, total_rows, combined_dataset.shape)
            combined_dataset.resize(total_rows, axis=0)
            combined_dataset[where_to_start_appending:total_rows, :] = data
            where_to_start_appending = total_rows
    if training_or_testing == 'testing':
        label_path = os.path.join(data_root_path, dataset,
                                  'testing_frames/{}_all_gt.txt'.format(dataset))
        label = np.loadtxt(label_path)
        combined_dataset = file_combined.create_dataset('label', data=label)
        print('{} test label successfully saved'.format(dataset))

    file_combined.close()

def get_label(image_value): # return frame label, input--image_value: ndarrary
    for pixel in image_value:
        if pixel != 0:
            label = 1
            return label
    label = 0
    return label

def generate_label(data_root_path, dataset):
    """generate label via pixel level"""
    total_frame_num = 0
    gt_path = os.path.join(data_root_path, dataset, 'testing_frames')
    gt_save_path = os.path.join(gt_path, '{}_test_gt'.format(dataset))
    if not os.path.isdir(gt_save_path): os.makedirs(gt_save_path, exist_ok=True)
    f = open(os.path.join(gt_path, 'testing_list_gt.txt'.format(dataset)), 'r')
    for video_count, file in enumerate(get_data_files(f)): # load every gt_folder
        folder_path = os.path.join(gt_path, file)
        image_list = [name for name in os.listdir(folder_path) if name.endswith('.bmp')]
        num_image = len(image_list)
        video_gt = np.zeros(shape=[num_image])
        for n, image in enumerate(sorted(image_list)): # load every image in folder
            if os.path.splitext(image)[1] == '.bmp':
                image_path = os.path.join(gt_path, file, image)
                image_value = imread(image_path, as_gray=False).ravel()
                label = get_label(image_value) # get label of every frame
                video_gt[n] = label
                total_frame_num += 1
        print(total_frame_num)
        np.savetxt(os.path.join(gt_save_path, '{}_testing_frames_Tset{:03d}.txt'.format(dataset, video_count+1)), video_gt)
        print('{} finish'.format(file))
    # concat all small labels into one txt
    #label_txt = open(os.path.join(gt_path, 'all_frame_gt.txt'), 'w')
    start_idx = end_index = 0
    all_gt = np.zeros(shape=total_frame_num)
    for file in os.listdir(gt_save_path):
        file_path = os.path.join(gt_save_path, file) # open every txt
        print('processing {}'.format(file))
        part_label = np.loadtxt(file_path)
        frame_num = len(part_label)
        end_index += frame_num
        all_gt[start_idx:end_index] = part_label
        start_idx = end_index
        print(end_index)
    np.savetxt(os.path.join(gt_path, '{}_all_gt.txt'.format(dataset)), all_gt)

def generate_label_index(data_root_path, dataset):
    """generate testing label via groundtruth index"""
    save_path = os.path.join(data_root_path, dataset, 'testing_frames/{}_test_gt'.format(dataset))
    if not os.path.isdir(save_path): os.makedirs(save_path, exist_ok=True)
    for num, video in enumerate(sorted(os.listdir(os.path.join(data_root_path, dataset, 'testing_frames/groundtruths_index')))):
        label_path = os.path.join(data_root_path, dataset, 'testing_frames/groundtruths_index', video)
        gt_vid = get_gt_vid(label_path)
        np.savetxt(os.path.join(save_path, '{}_testing_frames_Test{:03d}.txt'.format(dataset, num+1)), gt_vid)
        print('{} saved'.format(video))

def get_gt_vid(label_path):
    gt_vid_raw = np.loadtxt(label_path)
    gt_vid = np.zeros((200,))

    try:
        for event in range(gt_vid_raw.shape[1]):
            start_idx = int(gt_vid_raw[0, event])
            end_idx = int(gt_vid_raw[0, event]) + 1
            gt_vid[start_idx: end_idx] = 1

    except IndexError:
        start_idx = int(gt_vid_raw[0])
        end_idx = int(gt_vid_raw[1])
        gt_vid[start_idx: end_idx] = 1

    return gt_vid

def load_h5(h5_filename, train_or_eval, percentage):
    """percentage: how much data to train"""
    f = h5py.File(h5_filename)
    file_size = f['data'].shape[0]
    if train_or_eval == 'train':
        #data = f['data'][:int(file_size*percentage), :]
        data = f['data'][:]
    else:
        data = f['data'][int(file_size*percentage):, :]
    return data

def loadDataFile(h5_filename, train_or_eval, percentage):
    print('data loading start')
    return load_h5(h5_filename, train_or_eval, percentage)

def load_test_h5(h5_filename):
    f = h5py.File(h5_filename)
    file_size = f['data'].shape[0]
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def loadTestDataFile(h5_filename):
    return load_test_h5(h5_filename)

def split(dataset, percentage):
    return dataset[: int(percentage * dataset.shape[0])], dataset[int(percentage * dataset.shape[0]) + 1:] 
if __name__ == "__main__":
    #pixel_mean('UCSDped2', data_root_path)
    #image_subtract('UCSDped2', data_root_path)
    #combine_dataset('UCSDped1', data_root_path, 'training')
    #combine_dataset('UCSDped2', data_root_path, 'training')
    #combine_dataset('UCSDped1', data_root_path, 'testing')
    #combine_dataset('UCSDped2', data_root_path, 'testing')
    #generate_label(data_root_path, 'UCSDped2')
    #generate_label_index(data_root_path, 'UCSDped1')
    build_h5('UCSDped1', data_root_path, 'testing')
    build_h5('UCSDped2', data_root_path, 'testing')