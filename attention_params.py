import os
import tensorflow as tf

class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'



    # training
    batch_size = 20  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 10  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks  6 todo
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

FLAGS = tf.app.flags.FLAGS
root_dir = os.path.abspath('.') + '/data'
print('root data directory: ', root_dir)
# vocab_file = root_dir + '/vocabulary.txt'
# train_file = root_dir + '/training.txt'
# evaluation_file = root_dir + '/evaluation.txt'

tf.app.flags.DEFINE_string('input_training_data_path', root_dir + '/training.txt', 'training data path')
tf.app.flags.DEFINE_string('input_validation_data_path', root_dir + '/evaluation.txt', 'validation data path')
tf.app.flags.DEFINE_string('input_previous_model_path', root_dir + '/finalmodel.ckpt', 'path of previous model')
tf.app.flags.DEFINE_string('output_model_path', root_dir + '/finalmodel.ckpt', 'path to save model')
tf.app.flags.DEFINE_string('log_dir', root_dir + '/log_folder', 'folder to save checkpoints')
tf.app.flags.DEFINE_string('vocab_path', root_dir + '/vocabulary.txt', 'path of vocab dict')

tf.app.flags.DEFINE_integer('embedding_size', 1024, 'the dense embedding layer size')  # if = -1, then one-hot embedding
tf.app.flags.DEFINE_integer('win_size', 3, 'window size of convolution')
tf.app.flags.DEFINE_integer('conv_size', 1024, 'the convolution and max pooling layer size(kernal numbers)')
tf.app.flags.DEFINE_integer('dense_size', 128, 'the fully connect dense layer size')
tf.app.flags.DEFINE_bool('share_weight', True, 'whether to share weight between query and doc network')
tf.app.flags.DEFINE_integer('log_frequency', 100, 'log frequency')
tf.app.flags.DEFINE_integer('checkpoint_frequency', 1000, 'steps to save checkpoint')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'num of epochs to train')
tf.app.flags.DEFINE_integer('train_batch_size', 256, 'batch size of training procedure')     # **tune 256
tf.app.flags.DEFINE_integer('eval_batch_size', 128, 'batch size when evaluation')
tf.app.flags.DEFINE_integer('max_length', 50, 'query will be truncated if token count is larger than max_length')  # todo 256
tf.app.flags.DEFINE_integer('num_threads', 2, 'read thread for reading training data')
tf.app.flags.DEFINE_integer('negative_sample', 50, 'negative sample count')
tf.app.flags.DEFINE_float('softmax_gamma', 10.0, 'softmax gamma for loss function')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'which optimizer to use')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate to train the model')
tf.app.flags.DEFINE_bool('enable_early_stop', False, 'whether to use early stop')
tf.app.flags.DEFINE_integer('early_stop_steps', 30, 'How many bad checks to trigger early stop')

tf.app.flags.DEFINE_integer('training_steps', 100000, '')
tf.app.flags.DEFINE_float('loss_margin', 0.1, '')	# ** tune
tf.app.flags.DEFINE_integer('eval_every', 50, '')
tf.app.flags.DEFINE_integer('num_blocks', 6, '')        # todo 6
tf.app.flags.DEFINE_integer('num_heads', 8, '')
tf.app.flags.DEFINE_integer('validation_size', 50, '')
tf.app.flags.DEFINE_integer('evaluation_size', 100, '')


