import tensorflow as tf
import os
import sys
import numpy as np
from scipy.misc import imresize
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.join(ROOT_DIR, 'test_log'))
PRED_DIR = os.path.join(ROOT_DIR, 'pred')
if not os.path.isdir(LOG_DIR): os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.isdir(PRED_DIR): os.makedirs(PRED_DIR, exist_ok=True)
sys.path.append(ROOT_DIR)
import model, processing
MODEL_PATH = os.path.join(ROOT_DIR, 'log_4/model.ckpt')
LOG_FOUT = open(os.path.join(LOG_DIR, 'test_log.txt'), 'w')

DATA_DIR = '/disk/soya/datasets/UCSDped'
LABEL_DIR = '/disk/soya/datasets/UCSDped/h5_data/labels'
TEST_FILES = os.path.join(DATA_DIR, 'h5_data/test_2')

GPU_INDEX = 1
BATCH_SIZE = 1

def output_log(out_str):
    print(out_str)
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 99999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_frr + item_far) / 2
    return eer

def calc_auc_overall(pred_savepath ,gt_savepath):
    all_gt = []
    all_pred = []
    for video in sorted(os.listdir(gt_savepath)):
        gt_path = os.path.join(gt_savepath, video)
        part_gt = np.loadtxt(gt_path) # 200x1
        all_gt.append(part_gt)

    for video in sorted(os.listdir(pred_savepath)):
        pred_path = os.path.join(pred_savepath, video)
        part_pred = np.load(pred_path)
        all_pred.append(part_pred)

    all_gt = np.asarray(all_gt)
    #all_gt = np.squeeze(all_gt, axis=0)
    print(all_gt.shape)
    all_gt = np.concatenate(all_gt).ravel().astype(int)
    all_pred = np.asarray(all_pred)
    print(all_pred.shape)
    all_pred = np.concatenate(all_pred).ravel()

    auc = roc_auc_score(all_gt, all_pred)
    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=0)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)
    return auc, eer

def get_volume_score(pred, gt, timestep): # output: 200x1
    file_size = pred.shape[0]
    volume_cost = np.zeros((file_size,))
    print(len(volume_cost))
    for i in range(file_size):
        volume_cost[i] = np.linalg.norm(pred[i] - gt[i]) # 191x1
    raw_cost = imresize(np.expand_dims(volume_cost, axis=1), ((file_size + timestep - 1), 1)) # 200x1
    abnormal_score = (raw_cost - np.min(raw_cost)) / max(raw_cost) # (0, 1) normalization
    regularity_score = 1 - abnormal_score
    print(regularity_score)
    return regularity_score

def get_frame_score(pred, gt):
    """pred input: ndarray, size: vol_numx10x227x227
       output regularity score: ndarray, size: vol_numx10"""
    file_size = pred.shape[0] # number of volume
    pixel_loss = (gt - pred)**2 # vol_numx10x227x227
    frame_loss = np.sqrt(np.sum(np.sum(pixel_loss, axis=2), axis=2)) # vol_numx10
    max_score = np.max(frame_loss, axis=1) # vol_numx1
    min_score = np.min(frame_loss, axis=1)
    regularity_score = np.zeros_like(frame_loss)
    for num, volume in enumerate(frame_loss):
        regularity_score[num] = 1 - ((volume - min_score[num]) / max_score[num])
    return regularity_score


def test():
    with tf.Graph().as_default():
        #with tf.device('/gpu:'+str(GPU_INDEX)):
        origin_volume_pl = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 10, 227, 227))
        #label_pl = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, None))
        is_training_pl = tf.placeholder(dtype=tf.bool, shape=())

        decoded = model.get_model(origin_volume_pl, is_training_pl)
        saver = tf.train.Saver()

            # create a session
            #config = tf.ConfigProto()
            #config.gpu_options.all_growth = True
            #config.allow_soft_placement = True
            #config.log_device_placement = True
        sess = tf.Session()

            # restore variables from trained model
            # loading trained model
        saver.restore(sess, MODEL_PATH)
        output_log('model restored from %s' % (MODEL_PATH))

        ops = {'origin_volume_pl': origin_volume_pl,
                   'is_training_pl': is_training_pl,
                   'decoded': decoded}

        eval_one_epoch(sess, ops)
        auc, eer = calc_auc_overall(os.path.join(PRED_DIR, 'score'), LABEL_DIR)
        output_log('Overall auc = {:.2f}%, Overall eer = {:.2f}%'.format(auc*100, eer*100))

def eval_one_epoch(sess, ops):
    is_training = False
    score_dir = os.path.join(PRED_DIR, 'score')
    pred_dir = os.path.join(PRED_DIR, 'pred')
    if not os.path.isdir(score_dir): os.makedirs(score_dir, exist_ok=True)
    if not os.path.isdir(pred_dir): os.makedirs(pred_dir, exist_ok=True)

    for num, test_file in enumerate(sorted(os.listdir(TEST_FILES))): # test file: one video
        output_log('----'+test_file+'----')
        current_data = processing.loadDataFile(os.path.join(TEST_FILES, test_file), 'train', 1)
        print(current_data.shape)

        file_size = current_data.shape[0]
        print(file_size)
        num_batches = file_size // BATCH_SIZE
        all_pred = np.zeros_like(current_data)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['origin_volume_pl']: current_data[start_idx: end_idx, :],
                         ops['is_training_pl']: is_training}
            pred_val = sess.run([ops['decoded']], feed_dict=feed_dict)
            # save predict result into a list
            all_pred[start_idx:end_idx, :] = pred_val[0]

        np.save(os.path.join(pred_dir, '{}_pred.npy'.format(test_file)), all_pred)
        pred_score = get_volume_score(all_pred, current_data, 10)
        np.save(os.path.join(score_dir, '{}_score.npy'.format(test_file)), pred_score)
        output_log('predict result of {} saved.'.format(test_file))

if __name__ == '__main__':
    test()
    LOG_FOUT.close()