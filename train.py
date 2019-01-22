import tensorflow as tf
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'log_2_1')
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
sys.path.append(ROOT_DIR)
import model, processing
LOG_FOUT = open(os.path.join(LOG_DIR, 'train_log.txt'), 'w')
DATA_DIR = '/disk/soya/datasets/UCSDped'
MODEL_PATH = os.path.join(ROOT_DIR, 'log_2_1')
GPU_INDEX = 0

BATCH_SIZE = 64
MAX_EPOCH = 200
BASE_LEARNING_RATE = 0.1

TRAIN_FILES = os.path.join(DATA_DIR, 'h5_data/train')
TEST_FILES = os.path.join(DATA_DIR, 'h5_data/test')

def output_log(out_str):
    print(out_str)
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, batch*BATCH_SIZE, 200000, 0.0005, staircase=True)
    return learning_rate

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False

            origin_volume_pl = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 10, 227, 227))
            is_training_pl = tf.placeholder(dtype=tf.bool, shape=())
            batch = tf.Variable(0, trainable=False)

            # get model and loss
            decoded = model.get_model(origin_volume_pl, is_training_pl)
            loss = model.get_loss_L2(decoded, origin_volume_pl)
            loss_summary = tf.summary.scalar('loss', loss)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # get training operator
            learning_rate = get_learning_rate(batch)
            lr_summary = tf.summary.scalar('learning_rate', learning_rate)

            sess = tf.Session(config=config)

            
            optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            init = tf.global_variables_initializer() # init variables
            sess.run(init, {is_training_pl: True})
            train_summary_op = tf.summary.merge([loss_summary, lr_summary])
            train_summary_dir = os.path.join(LOG_DIR, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            
            dev_summary_op = tf.summary.merge([loss_summary, lr_summary])
            dev_summary_dir = os.path.join(LOG_DIR, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            eval_summary_op = tf.summary.merge([loss_summary, lr_summary])
            eval_summary_dir = os.path.join(LOG_DIR, "summaries", "eval")
            eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)
            # saving all training variables
            saver = tf.train.Saver()

        # create session and set config
        
        def dev_step(x_batch, writer = None, summary_op = None):
            feet_dict = {ops['origin_volume_pl']: x_batch,
                         ops['is_training_pl']: True}
            loss, _, summary, step = sess.run([ops['loss'], ops['train_op'], summary_op, ops['step']], feed_dict=feet_dict)
            print('step: {}  loss: {:g}'.format(step, loss))
            if writer:
                    writer.add_summary(summary, step)
        def train_step(x_batch):
            feet_dict = {ops['origin_volume_pl']: x_batch,
                         ops['is_training_pl']: True}
            loss, _, summary, step = sess.run([ops['loss'], ops['train_op'], train_summary_op, ops['step']], feed_dict=feet_dict)
            print('step: {}  loss: {:g}'.format(step, loss))
            train_summary_writer.add_summary(summary, step)

        # loading trained model
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading tuned variables from %s' % (MODEL_PATH))

        # placeholder and option
        ops = {'origin_volume_pl':origin_volume_pl,
               'is_training_pl':is_training_pl,
               'decoded':decoded,
               'loss':loss,
               'merged': train_summary_op,
               'step':batch,
               'train_op':train_op}
        fn = os.listdir(TRAIN_FILES)[0]
        output_log('----'+str(fn)+'----')
        # loading training data from h5
        current_data = processing.loadDataFile(os.path.join(TRAIN_FILES, fn), 'train', 0.9)
        eval_data = processing.loadDataFile(os.path.join(TEST_FILES, 'UCSDped1_testing.h5'), 'test', 1)
        train_data, dev_data = processing.split(current_data, 0.8)
        print("data loading finished.")
        # file_size = current_data.shape[0]
        dev_every = 1000
        for epoch in range(MAX_EPOCH):
            output_log('***EPOCH %03d***' %(epoch))
            sys.stdout.flush() # update output
            num_batches = train_data.shape[0] // BATCH_SIZE
            for batch_index in range(num_batches):
                start_idx = batch_index * BATCH_SIZE
                end_idx = (batch_index+1) * BATCH_SIZE
                train_step(current_data[start_idx:end_idx, :])
                current_step = tf.train.global_step(sess, global_step)
                if current_step % dev_every == 0:
                    print('\nEvaluation')
                    dev_step(dev_data, dev_summary_writer, dev_summary_op)
            #eval_one_epoch(sess, ops, eval_writer)

            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                output_log("Model saved in file: %s" %(save_path))
                print('\nTesting')
                dev_step(eval_data, eval_summary_writer, eval_summary_op)


        
def train_one_epoch(sess, ops, train_writer, dev_writer):
    is_training = True
    for fn in os.listdir(TRAIN_FILES):
        output_log('----'+str(fn)+'----')
        # loading training data from h5
        current_data = processing.loadDataFile(os.path.join(TRAIN_FILES, fn), 'train', 0.9)
        print("data loading finished.")
        # file_size = current_data.shape[0]
        train_data, dev_data = processing.split(current_data, 0.8)
        num_batches = train_data.shape[0] // BATCH_SIZE
        loss_sum = 0
        total_seen = 0
        dev_per_step = 1000

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

        #output_log('mean loss: %f' % (loss_sum/float(num_batches)))
        output_log('mean loss: %f' % (loss_sum/float(total_seen)))

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
