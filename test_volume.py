import numpy as np
import os
import h5py


test_data_root = '/home/soya/datasets/UCSDped/UCSDped1/training_frames'
test_data = '/home/soya/datasets/UCSDped/UCSDped1/training_frames/training_frames_Train001.npy'

def build_volume(file, stride=1, time_length=10):
    data_frames = np.load(file)
    data_frames = data_frames.reshape(-1, 227, 227)  # 200x227x227
    num_frames = data_frames.shape[0]

    data_only_frames = np.zeros((num_frames-stride*(time_length-1), time_length, 227, 227)).astype('float64')
    i = 0
    vol = 0
    while (i + stride*(time_length-1)+1 <= num_frames):
        temp_data = data_frames[i : i + stride * (time_length - 1) + 1 : stride]
        data_only_frames[vol] = temp_data
        vol += 1
        i += 1
    return data_only_frames

def build_h5(dataset, data_root_path, h5_path, training_or_testing):
    training_frame_path = os.path.join(data_root_path, dataset, '{}_frames'.format(training_or_testing))
    h5_data_path = os.path.join(data_root_path, dataset, '{}_frames'.format(training_or_testing), h5_path)
    if not os.path.isdir(h5_data_path): os.makedirs(h5_data_path, exist_ok=True)
    count = 0
    for file in os.listdir(training_frame_path):
        if file.endswith('.npy'):
            data_name = os.path.join(training_frame_path, file)
            print(data_name)
            # data concatenate
            volume_stride_1 = build_volume(data_name, stride=1) # 191x10x227x227
            volume_stride_2 = build_volume(data_name, stride=2) # 182x10x227x227
            volume_stride_3 = build_volume(data_name, stride=3) # 173x10x227x227
            volume_concatenate = np.concatenate((volume_stride_1, volume_stride_2, volume_stride_3), axis=0) # 546x10x227x227
            count += 1
            with h5py.File(os.path.join(h5_path, '{}_frames_h5_{:03d}'.format(training_or_testing, count))) as f:
                if training_or_testing == 'training':
                    np.random.shuffle(volume_concatenate) # shuffle training data
                f['data'] = volume_concatenate

if __name__ == '__main__':
    test_volume_1 = build_volume(test_data, stride=3)
    test_volume_2 = build_volume(test_data, stride=2)
    test_volume_3 = build_volume(test_data, stride=1)
    volume_concatenate = np.concatenate((test_volume_1,test_volume_2,test_volume_3), axis=0)
    print(volume_concatenate.shape)