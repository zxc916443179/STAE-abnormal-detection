import os
import sys
import glob

flog_path = '/media/soya/Newsmy/datasets/UCSDped/UCSDped2/testing_frames'
flog = open(os.path.join(flog_path, 'testing_list_gt.txt'), 'w')

def output_log(out_str):
    print (out_str)
    flog.write(out_str+'\n')
    flog.flush()

def load_txt(flog_path):
    for i in sorted(os.listdir(flog_path)):
        path = os.path.join(flog_path, i)
        if os.path.isdir(path):
            if i.startswith('.'): continue
            elif i.endswith('_gt'): output_log(i)
            else: continue

def load_h5_txt(flog_path, training_or_testing):
    flog = open(os.path.join(flog_path, '{}_frames_loadh5.txt'.format(training_or_testing)), 'w')
    path = os.path.join(flog_path, '{}_frames'.format(training_or_testing))
    for i in sorted(os.listdir(path)):
        if i.endswith('.npy') and i.startswith('{}'.format(training_or_testing)):
            output_log(i)

if __name__ == '__main__':
    #load_h5_txt(flog_path, 'training')
    #load_h5_txt(flog_path, 'testing')
    load_txt(flog_path)