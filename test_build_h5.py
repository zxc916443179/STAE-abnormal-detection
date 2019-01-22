import numpy as np
import h5py
import os


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