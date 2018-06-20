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
    def train(self):
        self.max_acc = 1
        self.is_training = True
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
            # input_k = tf.placeholder(tf.int32, shape=(None, FLAGS.max_length))
            # input_v = tf.placeholder(tf.int32, shape=(None, FLAGS.max_length))
            q_encode = models.embed(inputs=input_q, vocab_size=vocab_size+1,num_units=hp.hidden_units)          # embedding size plus 1 for padding
            ap_encode = models.embed(inputs=input_ap, vocab_size=vocab_size+1,num_units=hp.hidden_units)
            an_encode = models.embed(inputs=input_an, vocab_size=vocab_size+1,num_units=hp.hidden_units)
            # k_encode = models.embed(input_k, vocab_size=vocab_size,num_units=hp.hidden_units)
            # v_encode = models.embed(input_v, vocab_size=vocab_size,num_units=hp.hidden_units)
            # apply dropout
            q_encode = tf.layers.dropout(q_encode, hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            ap_encode = tf.layers.dropout(ap_encode, hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            an_encode = tf.layers.dropout(an_encode, hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            # k_encode = tf.layers.dropout(k_encode, hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            # v_encode = tf.layers.dropout(v_encode, hp.dropout_rate, training=tf.convert_to_tensor(self.is_training))


            # multihead blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    q_encode = models.multihead_attention(query=q_encode, key=q_encode, value=q_encode,
                                                          num_heads=hp.num_heads,
                                                          is_training=tf.convert_to_tensor(self.is_training),
                                                          dropout_rate=hp.dropout_rate,
                                                          mask_future=False)
                    q_encode = models.feed_forward(q_encode, units=[hp.hidden_units * 4, hp.hidden_units])
                    ap_encode = models.multihead_attention(query=ap_encode, key=ap_encode, value=ap_encode,
                                                          num_heads=hp.num_heads,
                                                          is_training=tf.convert_to_tensor(self.is_training),
                                                          dropout_rate=hp.dropout_rate,
                                                          mask_future=False)
                    ap_encode = models.feed_forward(ap_encode, units=[hp.hidden_units * 4, hp.hidden_units])
                    an_encode = models.multihead_attention(query=an_encode, key=an_encode, value=an_encode,
                                                           num_heads=hp.num_heads,
                                                           is_training=tf.convert_to_tensor(self.is_training),
                                                           dropout_rate=hp.dropout_rate,
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
                # meta_path = FLAGS.output_model_path + '/step6500_loss0.0_trainAcc1.0_evalAcc0.36.meta'
                # model_path = FLAGS.output_model_path + '/step6500_loss0.0_trainAcc1.0_evalAcc0.36'
                # saver = tf.train.import_meta_graph(meta_path)
                # print('graph imported')
                # saver.restore(session, model_path)
                # print('variables restored!')

                # Load pre-trained model
                ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(session, ckpt.model_checkpoint_path)
                    print("Load Model From ", ckpt.model_checkpoint_path)
                else:
                    print("No model found")

                print("Begin to train model.")
                max_acc = 0
                for step in range(FLAGS.training_steps):
                    train_data_batch = data_processor.next_batch_train_random(FLAGS.train_batch_size)
                    train_q_vec_b, train_q_vec_len_b, train_d_vec_b, train_d_vec_len_b, train_dneg_vec_b, train_dneg_vec_len_b = train_data_batch
                    feed_dict = {input_q:train_q_vec_b,
                                 input_ap:train_d_vec_b,
                                 input_an:train_dneg_vec_b}
                    _, loss_avg_, accuracy_, step_ = session.run([train_op, loss_avg, accuracy, global_step], feed_dict=feed_dict)
                    print('=' * 10 + 'step{}, loss_avg = {}, acc={}'.format(step_, loss_avg_, accuracy_))  # loss for all batches
                    if step_ % FLAGS.eval_every == 0:
                        print('\n============================> begin to test ')
                        eval_size = FLAGS.validation_size
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
                            # cos_pos_, cos_neg_, accuracy_ = sess.run([cos_12, cos_13, accuracy], feed_dict)
                            # data_processor.saveFeatures(cos_pos_, cos_neg_, test_loss_, accuracy_)
                            return correct_flag

                        def evaluate(eval_size):
                            correct_num = int(0)
                            for i in range(eval_size):
                                print('evaluation step %d '%i)
                                batches = data_processor.loadValData_step(vocab, vocab_size,
                                                                          FLAGS.input_validation_data_path,
                                                                          FLAGS.max_length, eval_size)  # batch_size*seq_len
                                # 显示/保存测试数据
                                # save_test_data(batch_y1, batch_y2, label_list)
                                batch_y1, batch_y2, label_list = batches[i]
                                correct_flag = test_step(batch_y1, batch_y2, batch_y2, label_list, session)
                                correct_num += correct_flag
                            print('correct_num', correct_num)
                            acc = correct_num / float(eval_size)
                            return acc

                        self.is_training = False
                        acc_ = evaluate(eval_size=eval_size)
                        self.is_training = True
                        print(
                            '--------The test result among the test data sets: acc = {0}, test size = {1}----------'.format(
                                acc_, eval_size))
                        if acc_ >= max_acc:
                            max_acc = acc_
                        # # acc = test_for_bilstm.test()
                            path = saver.save(session, FLAGS.output_model_path + '/step' + str(step_) + '_loss' + str(
                                loss_avg_) + '_trainAcc' + str(accuracy_) + '_evalAcc' + str(acc_))
                            saver.export_meta_graph(
                                FLAGS.output_model_path + '/meta_' + 'step' + str(step_) + '_loss' + str(
                                    loss_avg_) + '_trainAcc' + str(accuracy_) + '_evalAcc' + str(acc_))
                            print("Save checkpoint(model) to {}".format(path))

if __name__ == '__main__':
    # Load vocabulary
    # Construct graph
    g = Graph();
    print("Graph loaded")
    g.train()

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
