import tensorflow as tf
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'log')
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
sys.path.append(ROOT_DIR)
import model, processing
LOG_FOUT = open(os.path.join(LOG_DIR, 'train_log.txt'), 'w')
DATA_DIR = '/disk/soya/datasets/UCSDped'
GPU_INDEX = 0

BATCH_SIZE = 64
MAX_EPOCH = 50
BASE_LEARNING_RATE = 0.001

TRAIN_FILES = os.path.join(DATA_DIR, 'h5_data/train')
TEST_FILES = os.path.join(DATA_DIR, 'h5_data/test')

def output_log(out_str):
    print(out_str)
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, batch*BATCH_SIZE, 200000, 0.7, staircase=True)
    return learning_rate

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            origin_volume_pl = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 10, 227, 227))
            is_training_pl = tf.placeholder(dtype=tf.bool, shape=())
            batch = tf.Variable(0, trainable=False)

            # get model and loss
            decoded = model.get_model(origin_volume_pl, is_training_pl)
            loss = model.get_loss_L2(decoded, origin_volume_pl)
            tf.summary.scalar('loss', loss)

            # get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # saving all training variables
            saver = tf.train.Saver()

        # create session and set config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        eval_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        init = tf.global_variables_initializer() # init variables

        sess.run(init, {is_training_pl: True})

        # loading trained model
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading tuned variables from %s' % (LOG_DIR))

        # placeholder and option
        ops = {'origin_volume_pl':origin_volume_pl,
               'is_training_pl':is_training_pl,
               'decoded':decoded,
               'loss':loss,
               'merged':merged,
               'step':batch,
               'train_op':train_op}

        for epoch in range(MAX_EPOCH):
            output_log('***EPOCH %03d***' %(epoch))
            sys.stdout.flush() # update output

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, eval_writer)

            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                output_log("Model saved in file: %s" %(save_path))

def train_one_epoch(sess, ops, train_writer):
    is_training = True
    for fn in os.listdir(TRAIN_FILES):
        output_log('----'+str(fn)+'----')
        # loading training data from h5
        current_data = processing.loadDataFile(os.path.join(TRAIN_FILES, fn), 'train', 0.9)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        loss_sum = 0
        total_seen = 0

        for batch_index in range(num_batches):
            start_idx = batch_index * BATCH_SIZE
            end_idx = (batch_index+1) * BATCH_SIZE

            feet_dict = {ops['origin_volume_pl']: current_data[start_idx:end_idx, :],
                         ops['is_training_pl']: is_training}
            loss, decoded, _, summary, step = sess.run([ops['loss'], ops['decoded'], ops['train_op'],
                                                 ops['merged'], ops['step']], feed_dict=feet_dict)
            total_seen += BATCH_SIZE
            loss_sum += loss
            train_writer.add_summary(summary, step)

        output_log('mean loss: %f' % (loss_sum/float(num_batches)))

def eval_one_epoch(sess, ops, eval_writer):
    is_training = False
    for fn in os.listdir(TRAIN_FILES):
        output_log('----eval_'+str(fn)+'----')
        # loading validation data from h5
        current_eval_Data = processing.loadDataFile(os.path.join(TRAIN_FILES, fn), 'eval', 0.9)
        file_size = current_eval_Data.shape[0]
        num_batches = file_size // BATCH_SIZE
        loss_sum = 0
        total_seen = 0

        for batch_index in range(num_batches):
            start_idx = batch_index * BATCH_SIZE
            end_idx = (batch_index+1) * BATCH_SIZE

            feed_dict = {ops['origin_volume_pl']: current_eval_Data[start_idx:end_idx, :],
                         ops['is_training_pl']: is_training}
            loss, decoded, _, summary, step = sess.run([ops['loss'], ops['decoded'], ops['train_op'],
                                                        ops['merged'], ops['step']], feed_dict=feed_dict)
            total_seen += BATCH_SIZE
            loss_sum += loss
            eval_writer.add_summary(summary, step)

        output_log('eval mean loss: %f' % (loss_sum/float(num_batches)))

if __name__ == '__main__':
    train()
    LOG_FOUT.close()