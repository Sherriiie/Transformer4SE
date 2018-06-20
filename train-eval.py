import tensorflow as tf
from attention_params import Hyperparams as hp
from attention_params import FLAGS
from attention_model import AttentionModel
from attention_datapro import DataProcessor
import tqdm
import numpy as np

class Graph():
    def __int__(self, is_training=True):
        self.is_training = True
        self.max_acc = 0
        return
    def eval(self):
        self.max_acc = 1
        self.is_training = False
        with tf.Graph().as_default():
            data_processor = DataProcessor()

            vocab_size = data_processor.get_vocabulary_size(FLAGS.vocab_path)
            vocab, revocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
            data_processor.get_init(FLAGS.input_training_data_path, FLAGS.input_validation_data_path, vocab, vocab_size,
                                    FLAGS.max_length, revocab)
            models = AttentionModel()

            input_q = tf.placeholder(tf.int32, shape=(None, FLAGS.max_length), name="input_x1")     # FLAGS.train_batch_size
            input_ap = tf.placeholder(tf.int32, shape=(None, FLAGS.max_length))
            input_an = tf.placeholder(tf.int32, shape=(None, FLAGS.max_length))
            q_encode = models.embed(inputs=input_q, vocab_size=vocab_size+1,num_units=hp.hidden_units)          # embedding size plus 1 for padding
            ap_encode = models.embed(inputs=input_ap, vocab_size=vocab_size+1,num_units=hp.hidden_units)
            an_encode = models.embed(inputs=input_an, vocab_size=vocab_size+1,num_units=hp.hidden_units)

            # multihead blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    q_encode = models.multihead_attention(query=q_encode, key=q_encode, value=q_encode,
                                                          num_heads=hp.num_heads,
                                                          mask_future=False)
                    q_encode = models.feed_forward(q_encode, units=[hp.hidden_units * 4, hp.hidden_units])
                    ap_encode = models.multihead_attention(query=ap_encode, key=ap_encode, value=ap_encode,
                                                          num_heads=hp.num_heads,
                                                          mask_future=False)
                    ap_encode = models.feed_forward(ap_encode, units=[hp.hidden_units * 4, hp.hidden_units])
                    an_encode = models.multihead_attention(query=an_encode, key=an_encode, value=an_encode,
                                                           num_heads=hp.num_heads,
                                                           mask_future=False)
                    an_encode = models.feed_forward(an_encode, units=[hp.hidden_units * 4, hp.hidden_units])

                ## output layer
            with tf.name_scope('output_layer'):
                dims = q_encode.get_shape().as_list()
                q_encode = tf.reshape(q_encode, [-1, dims[1]*dims[2]])
                ap_encode = tf.reshape(ap_encode, [-1, dims[1]*dims[2]])
                an_encode = tf.reshape(an_encode, [-1, dims[1]*dims[2]])
                weight = tf.get_variable('output_weight', [q_encode.get_shape().as_list()[-1], hp.hidden_units])

                q_encode = tf.matmul(q_encode, weight)
                ap_encode = tf.matmul(ap_encode, weight)
                an_encode = tf.matmul(an_encode, weight)

            q_encode = models.vec_normalize(q_encode)
            ap_encode = models.vec_normalize(ap_encode)
            an_encode = models.vec_normalize(an_encode)


            ## calculate similarity and loss
            cos_12 = tf.reduce_sum(tf.multiply(q_encode, ap_encode), 1)  # wisely multiple vectors
            cos_13 = tf.reduce_sum(tf.multiply(q_encode, an_encode), 1)

            zero = tf.constant(0, shape=[FLAGS.train_batch_size], dtype=tf.float32)
            margin = tf.constant(FLAGS.loss_margin, shape=[FLAGS.train_batch_size], dtype=tf.float32)

            losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(cos_12, cos_13)))
            loss_sum = tf.reduce_sum(losses)
            loss_avg = tf.div(loss_sum, FLAGS.train_batch_size)
            correct = tf.equal(zero, losses)
            accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

            global_step = tf.Variable(0, name="global_step",
                                      trainable=False)  # The global step will be automatically incremented by one every time you execute　a train loop
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss_avg)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables())

            # session start point
            with tf.Session() as session:
                session.run(tf.local_variables_initializer())
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())

                # Load pre-trained model
                ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(session, ckpt.model_checkpoint_path)
                    print("Load Model From ", ckpt.model_checkpoint_path)
                else:
                    print("No model found and exit.")
                    exit()

                print('\n============================> begin to evaluate model. ')
                eval_size = FLAGS.evaluation_size

                def evaluate_all():
                    correct_num = int(0)
                    batches = data_processor.loadValData_step(vocab, vocab_size,
                                                              FLAGS.input_validation_data_path,
                                                              FLAGS.max_length, eval_size=0)  # batch_size*seq_len
                    for i in range(eval_size):
                        # 显示/保存测试数据
                        # save_test_data(batch_y1, batch_y2, label_list)
                        batch_y1, batch_y2, label_list = batches[i]
                        correct_flag = test_step(batch_y1, batch_y2, batch_y2, label_list, session)
                        correct_num += correct_flag
                        if (correct_flag == 1):
                            print('step %d ==== correct prediction' % i)
                        else:
                            print('step %d ==== wrong prediction' % i)
                    print('correct_num', correct_num)
                    acc = correct_num / float(eval_size)
                    return acc


                def test_step(input_y1, input_y2, input_y3, label_list, sess):
                    feed_dict = {
                        input_q: input_y1,
                        input_ap: input_y2,
                        input_an: input_y3}

                    correct_flag = 0
                    cos_12_ = sess.run(cos_12, feed_dict)
                    cos_max = max(cos_12_)
                    index_max = list(cos_12_).index(cos_max)
                    if label_list[index_max] == '1':
                        correct_flag = 1
                    return correct_flag

                def evaluate(eval_size):
                    correct_num = int(0)
                    batches = data_processor.loadValData_step(vocab, vocab_size,
                                                              FLAGS.input_validation_data_path,
                                                              FLAGS.max_length, eval_size)  # batch_size*seq_len
                    for i in range(eval_size):
                        # 显示/保存测试数据
                        # save_test_data(batch_y1, batch_y2, label_list)
                        batch_y1, batch_y2, label_list = batches[i]
                        correct_flag = test_step(batch_y1, batch_y2, batch_y2, label_list, session)
                        correct_num += correct_flag
                        if (correct_flag==1):
                            print('step %d ==== correct prediction' %i)
                        else:
                            print('step %d ==== wrong prediction' %i)
                    print('correct_num', correct_num)
                    acc = correct_num / float(eval_size)
                    return acc

                acc_ = evaluate(eval_size=eval_size)
                print(
                    '--------The test result among the test data sets: acc = {0}, test size = {1}----------'.format(
                        acc_, eval_size))
                exit()


if __name__ == '__main__':
    # Load vocabulary
    # Construct graph
    g = Graph();
    print("Graph loaded")
    g.eval()

    # # Start session
    # sv = tf.train.Supervisor(graph=g.graph,
    #                          logdir=hp.logdir,
    #                          save_model_secs=0)
    # with sv.managed_session() as sess:
    #     for epoch in range(1, hp.num_epochs + 1):
    #         if sv.should_stop(): break
    #         for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
    #             sess.run(g.train_op)
    #
    #         gs = sess.run(g.global_step)
    #         sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")
