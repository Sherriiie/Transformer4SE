import tensorflow as tf
# from attention_params import Hyperparams as hp
# from attention_model import AttentionModel as models
import random
import os
import numpy as np
from attention_params import FLAGS


class DataProcessor:
    """
    process cdssm model training/evaluation data
    """

    def __init__(self):
        self.data_path_train = ''
        self.data_path_eval = ''
        # self.data_path = data_path
        self.positive_pairs = []
        self.queries = []
        self.queries_vec = []
        self.answers = []
        self.answers_vec = []
        self.answers_neg_vec = []
        self.answers_neg_vec = []
        self.counter = 0  # for train
        self.total_item_count = 0  # positive pairs

        self.eval_positive_pairs = []
        self.eval_queries = []
        self.eval_queries_vec = []
        self.eval_answers = []
        self.eval_answers_vec = []

        self.eval_counter = 0  # for eval
        self.eval_total_item_count = 0
        # no dup in answers_neg
        # print('there are %d answers in neg no dup'%len(self.answers_neg))
        self.answers_neg = []
        self.answers_neg_vec = []
        # write answers_neg to file
        self.flag = True
        self.vocab_size = 0
        self.revocab_dict = {}

    def get_init(self, data_path_train, data_path_eval, vocab, vocab_size, max_length, revocab_dict):
        self.data_path_train = data_path_train
        self.data_path_eval = data_path_eval
        # self.data_path = data_path
        self.vocab_size = vocab_size
        self.counter = 0  # for train
        self.total_item_count = 0  # positive pairs
        self.read_train_data(vocab, vocab_size, max_length)  # read data into system
        self.eval_counter = 0  # for eval
        self.eval_total_item_count = 0
        self.read_eval_data(vocab, vocab_size, max_length)
        # no dup in answers_neg
        self.answers_neg = list(set(self.answers))
        print(
            'There are %d answers in neg no dup, writing into file <data/answers_evaluation.txt>' % len(
                self.answers_neg))
        # write answers_neg to file
        self.write_to_file(self.answers_neg)
        self.revocab_dict = revocab_dict

    def write_to_file(self, answers):
        file_write = open('data/answers_evaluation.txt', encoding='utf-8', mode='w')
        temp = [ent.replace(' ', '') for ent in answers]
        file_write.write('\n'.join(temp))
        file_write.flush()

    def read_train_data(self, vocab, vocab_size, max_length):
        with open(self.data_path_train, encoding='utf-8', mode='r') as file_reader:
            for line in file_reader:
                line = line.strip()
                self.positive_pairs.append(line)
                parts = line.split('\t')
                self.queries.append(parts[0])
                self.answers.append(parts[1])
                query = parts[0]
                ent = parts[1]
                query_vec = DataProcessor.vectorize(query, vocab, vocab_size)
                ent_vec = DataProcessor.vectorize(ent, vocab, vocab_size)
                if (len(query_vec) >= max_length):
                    query_vec = query_vec[0:max_length]
                else:
                    query_vec += [self.vocab_size] * (max_length - len(query_vec))
                if (len(ent_vec) >= max_length):
                    ent_vec = ent_vec[0:max_length]
                else:
                    ent_vec += [self.vocab_size] * (max_length - len(ent_vec))
                self.queries_vec.append(query_vec)
                self.answers_vec.append(ent_vec)
            self.total_item_count = len(self.positive_pairs)
            self.answers_neg = list(set(self.answers))
            for ent in self.answers_neg:
                ent_vec = DataProcessor.vectorize(ent, vocab, vocab_size)
                if (len(ent_vec) >= max_length):
                    ent_vec = ent_vec[0:max_length]
                else:
                    ent_vec += [self.vocab_size] * (max_length - len(ent_vec))
                self.answers_neg_vec.append(ent_vec)
            print('==== read training data into model finished')

    def read_eval_data(self, vocab, vocab_size, max_length):
        with open(self.data_path_eval, encoding='utf-8', mode='r') as file_reader:
            for line in file_reader:
                line = line.strip()
                self.eval_positive_pairs.append(line)
                parts = line.split('\t')
                self.eval_queries.append(parts[0])
                self.eval_answers.append(parts[1])
                eval_query = parts[0]
                eval_ent = parts[1]
                eval_query_vec = DataProcessor.vectorize(eval_query, vocab, vocab_size)
                eval_ent_vec = DataProcessor.vectorize(eval_ent, vocab, vocab_size)
                if (len(eval_query_vec) >= max_length):
                    eval_query_vec = eval_query_vec[0:max_length]
                else:
                    eval_query_vec += [self.vocab_size] * (max_length - len(eval_query_vec))
                if (len(eval_ent_vec) >= max_length):
                    eval_ent_vec = eval_ent_vec[0:max_length]
                else:
                    eval_ent_vec += [self.vocab_size] * (max_length - len(eval_ent_vec))
                self.eval_queries_vec.append(eval_query_vec)
                self.eval_answers_vec.append(eval_ent_vec)
            self.eval_total_item_count = len(self.eval_positive_pairs)
            print('==== read eval data finished, total count = ', self.eval_total_item_count)

    @staticmethod
    def get_vocabulary_size(vocab_file_path):
        """
        get vocabulary size
        """
        vocab_size = 0
        with open(vocab_file_path, encoding='utf-8', mode='rt') as vocab_file:
            for line in vocab_file:
                if line.strip():
                    vocab_size += 1
        return vocab_size

    @staticmethod
    def initialize_vocabulary(vocab_file_path):
        """
        load vocabulary from file
        """
        if os.path.exists(vocab_file_path):
            data_list = []
            with open(vocab_file_path, encoding='utf-8', mode='rt') as vocab_file:
                for line in vocab_file:
                    if line.strip():
                        data_list.append(line.strip())
            vocab_dict = dict([(x, y) for (y, x) in enumerate(data_list)])  # (char, index)
            revocab_dict = dict([(x, y) for (x, y) in enumerate(data_list)])  # (index, char)
            return vocab_dict, revocab_dict
        else:
            raise ValueError('Vocabulary file {} not found.'.format(vocab_file_path))

    @staticmethod
    def vectorize(text, vocab, default):
        """
        vectorize text to word ids based on vocab
        """
        return [vocab.get(word, default) for word in text.split(' ')]

    def devectorize(self, index, default):
        """
        vectorize text to word ids based on vocab
        """
        return [self.revocab_dict.get(idx, default) for idx in index]

    @staticmethod
    def create_tfrecord(data_file_path, tfrecords_file_path, vocab, is_training=True):
        """
        vectorize training data and convert to tfrecords
        """
        vocab_size = len(vocab)
        query_to_id = {}
        current_id = 0

        with open(data_file_path, encoding='utf-8', mode='rt') as reader, tf.python_io.TFRecordWriter(
                tfrecords_file_path) as writer:
            for line in reader:
                segs = line.strip().split("\t")
                if len(segs) < 2:
                    continue

                query = segs[0]
                doc = segs[1]
                query_vec = DataProcessor.vectorize(query, vocab, vocab_size)
                doc_vec = DataProcessor.vectorize(doc, vocab, vocab_size)

                if is_training:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'query_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=query_vec)),
                        'doc_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=doc_vec)),
                    }))
                else:
                    label = int(segs[2])
                    if not query in query_to_id:
                        query_to_id[query] = current_id
                        current_id += 1
                    query_id = query_to_id[query]
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'query_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[query_id])),
                        'query_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=query_vec)),
                        'doc_vec': tf.train.Feature(int64_list=tf.train.Int64List(value=doc_vec)),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }))

                writer.write(example.SerializeToString())

    @staticmethod
    def load_training_tfrecords(tfrecords_file_path, num_epochs, batch_size, max_length, num_threads):
        """
        load training data from tfrecords
        """
        filename_queue = tf.train.string_input_producer([tfrecords_file_path], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature_configs = {
            'query_vec': tf.VarLenFeature(dtype=tf.int64),
            'doc_vec': tf.VarLenFeature(dtype=tf.int64),
        }
        features = tf.parse_single_example(serialized_example, features=feature_configs)
        query_vec, query_vec_length = DataProcessor.parse_feature(features['query_vec'], max_length)
        doc_vec, doc_vec_length = DataProcessor.parse_feature(features['doc_vec'], max_length)
        print('query_vec', query_vec.get_shape(), query_vec_length.get_shape())
        print('doc_vec', doc_vec.get_shape())
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch([query_vec, query_vec_length, doc_vec, doc_vec_length],
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      num_threads=num_threads,
                                      allow_smaller_final_batch=True)

    def next_batch_train_random(self, batch_size):
        # Author: Sherrie
        b_query_vec = []
        b_query_length = []
        b_answer_vec = []
        b_answerneg_vec = []
        b_answer_length = []
        b_answerneg_length = []
        # b_label = []
        for i in range(batch_size):
            # if self.flag == True:
            #     self.flag = not self.flag
                b_query_vec.append(self.queries_vec[self.counter])
                b_query_length.append(len(self.queries_vec[self.counter]))
                b_answer_vec.append((self.answers_vec[self.counter]))
                b_answer_length.append(len(self.answers_vec[self.counter]))
                # b_label.append(1)
                self.counter += 1
                self.counter = self.counter % self.total_item_count
            # else:
            #     self.flag = not self.flag
            #     b_query_vec.append(self.queries_vec[self.counter])
            #     b_query_length.append(len(self.queries_vec[self.counter]))
                index = random.randint(0, self.total_item_count - 1)
                while (self.answers_vec[index] == self.answers_vec[self.counter]):
                    index = random.randint(0, self.total_item_count - 1)
                b_answerneg_vec.append((self.answers_vec[index]))
                b_answerneg_length.append(len(self.answers_vec[self.counter]))
                # b_label.append(0)
        return [b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_answerneg_vec, b_answerneg_length]

    def next_batch_evaluation(self):
        b_query_vec = []
        b_query_length = []
        b_answer_vec = []
        b_answer_length = []
        b_label = []
        answers_neg_vec = self.answers_neg_vec.copy()

        # add the postive sample first, then add all answers as negative, so the total number is +1
        b_query_vec.append(self.eval_queries_vec[self.eval_counter])
        b_query_length.append(len(self.eval_queries_vec[self.eval_counter]))
        b_answer_vec.append(self.eval_answers_vec[self.eval_counter])
        b_answer_length.append(len(self.eval_answers_vec[self.eval_counter]))
        b_label.append(1)
        if (len(answers_neg_vec) != 162):
            print('bug')
        # add all other answers as negative samples
        for i in range(len(answers_neg_vec)):
            b_query_vec.append(self.eval_queries_vec[self.eval_counter])
            b_query_length.append(len(self.eval_queries_vec[self.eval_counter]))
            b_answer_vec.append(answers_neg_vec[i])
            b_answer_length.append(len(answers_neg_vec[i]))
            b_label.append(1)
        # answers_neg.append(self.eval_answers[self.eval_counter])
        self.eval_counter += 1
        self.eval_counter = self.eval_counter % self.eval_total_item_count
        return [b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_label]

    def random_index(self, count):
        index = [i for i in range(self.eval_total_item_count)]
        random.shuffle(index)
        return index[:count]

    def random_batch_evaluation(self, count):
        batches = []
        # get the index of random
        random_index_list = self.random_index(count)
        for i in range(count):
            b_query_vec = []
            b_query_length = []
            b_answer_vec = []
            b_answer_length = []
            b_label = []
            answers_neg_vec = self.answers_neg_vec.copy()

            # add the postive sample first, then add all answers as negative, so the total number is +1
            # random_index = random.randint(0, self.eval_total_item_count-1)          # self.eval_counter
            random_index = random_index_list[i]
            b_query_vec.append(self.eval_queries_vec[random_index])
            b_query_length.append(len(self.eval_queries_vec[random_index]))
            b_answer_vec.append(self.eval_answers_vec[random_index])
            b_answer_length.append(len(self.eval_answers_vec[random_index]))
            b_label.append(1)
            if (len(answers_neg_vec) != 162):
                print('bug')
            # add all other answers as negative samples
            for i in range(len(answers_neg_vec)):
                b_query_vec.append(self.eval_queries_vec[random_index])
                b_query_length.append(len(self.eval_queries_vec[random_index]))
                b_answer_vec.append(answers_neg_vec[i])
                b_answer_length.append(len(answers_neg_vec[i]))
                b_label.append(1)
            batches.append([b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_label])
            # return [b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_label]
        return batches

    # no dup
    def random_batch_evaluation2(self, count):
        batches = []
        qa = []
        # get the index of random
        random_index_list = self.random_index(count)
        for i in range(count):
            b_query_vec = []
            b_query_length = []
            b_answer_vec = []
            b_answer_length = []
            b_label = []
            answers_neg_vec = self.answers_neg_vec.copy()

            # add the postive sample first, then add all answers as negative, so the total number is +1
            # random_index = random.randint(0, self.eval_total_item_count-1)          # self.eval_counter
            random_index = random_index_list[i]
            # b_query_vec.append(self.eval_queries_vec[random_index])
            # b_query_length.append(len(self.eval_queries_vec[random_index]))
            # b_answer_vec.append(self.eval_answers_vec[random_index])
            # b_answer_length.append(len(self.eval_answers_vec[random_index]))
            # b_label.append(1)
            if (len(answers_neg_vec) != 162):
                print('bug')
            # add all other answers as negative samples
            for i in range(len(answers_neg_vec)):
                b_query_vec.append(self.eval_queries_vec[random_index])
                b_query_length.append(len(self.eval_queries_vec[random_index]))
                b_answer_vec.append(answers_neg_vec[i])
                b_answer_length.append(len(answers_neg_vec[i]))
                b_label.append(1)
            batches.append([b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_label])
            qa.append([self.eval_queries_vec[random_index], self.eval_answers_vec[random_index]])
            # return [b_query_vec, b_query_length, b_answer_vec, b_answer_length, b_label]
        return batches, qa

    # get answer name by index
    def get_answer_by_index(self, answer_index):
        if 0 not in answer_index:
            return [self.answers_neg[idx - 1] for idx in answer_index]

    def get_answer_by_index2(self, answer_index):
        # if 0 not in answer_index:
        return [self.answers_neg[idx] for idx in answer_index]

    def get_char_by_index(self, index):
        return [self.initialize_vocabulary() for idx in index]

    @staticmethod
    def load_evaluation_tfrecords(tfrecords_file_path, batch_size, max_length):
        """
        load evaluation data from tfrecords
        """
        query_id_batch = []
        query_vec_batch = []
        doc_vec_batch = []
        label_batch = []
        cnt = 0
        for serialized_example in tf.python_io.tf_record_iterator(tfrecords_file_path):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            query_id = example.features.feature['query_id'].int64_list.value[0]
            query_vec = example.features.feature['query_vec'].int64_list.value
            doc_vec = example.features.feature['doc_vec'].int64_list.value
            label = example.features.feature['label'].int64_list.value[0]

            query_id_batch.append(query_id)
            query_vec_batch.append(query_vec)
            doc_vec_batch.append(doc_vec)
            label_batch.append(label)

            cnt += 1
            if cnt % batch_size == 0:
                query_vec_batch, query_vec_length_batch = DataProcessor.align_vector_batch(query_vec_batch,
                                                                                           max_length)
                doc_vec_batch, doc_vec_length_batch = DataProcessor.align_vector_batch(doc_vec_batch,
                                                                                       max_length)
                yield query_id_batch, query_vec_batch, query_vec_length_batch, doc_vec_batch, doc_vec_length_batch, label_batch

                query_id_batch = []
                query_vec_batch = []
                doc_vec_batch = []
                label_batch = []

        if query_id_batch:
            query_vec_batch, query_vec_length_batch = DataProcessor.align_vector_batch(query_vec_batch,
                                                                                       max_length)
            doc_vec_batch, doc_vec_length_batch = DataProcessor.align_vector_batch(doc_vec_batch, max_length)
            yield query_id_batch, query_vec_batch, query_vec_length_batch, doc_vec_batch, doc_vec_length_batch, label_batch

    @staticmethod
    def align_vector_batch(vector_batch, max_length):
        """
        align vector batch to max length
        """
        result_vector_batch = []
        result_vector_length_batch = []

        batch_max_length = max(len(vector) for vector in vector_batch)
        align_length = min(batch_max_length, max_length)

        for vector in vector_batch:
            result_vector, result_vector_length = DataProcessor.align_vector(vector, align_length)
            result_vector_batch.append(result_vector)
            result_vector_length_batch.append(result_vector_length)

        return result_vector_batch, result_vector_length_batch

    @staticmethod
    def align_vector(vector, max_length):
        """
        align vector to max length
        """
        vector_length = len(vector)
        if vector_length > max_length:
            vector_length = max_length
            vector = vector[:max_length]
        else:
            vector.extend([0 for _ in range(max_length - vector_length)])
        return vector, vector_length

    @staticmethod
    def parse_feature(feature, max_length):
        """
        deserialize feature
        """
        feature_length = tf.minimum(feature.dense_shape[0], tf.constant(max_length, tf.int64))
        print("parse_feature: feature.shape()", feature.get_shape())
        feature = tf.sparse_to_dense(sparse_indices=feature.indices[:max_length], output_shape=[max_length],
                                     sparse_values=feature.values[:max_length], default_value=0)
        return feature, feature_length

    def loadValData_step(self, vocab, vocab_size, file_path, sequence_length, size):
        batches = []
        input_x1 = []  # questions
        input_x2 = []  # positive answers
        labels = []  # negative answers
        # load datasets {q, [[ans,label]], ...}
        valDataSets = self.loadDataSets(vocab, vocab_size, file_path, sequence_length)
        # print('length of evaluation data: ', len(valDataSets.keys()))
        random_keys = random.sample(list(valDataSets), size)
        for random_key in random_keys:
            random_values = valDataSets[random_key]
            random_key_vec = DataProcessor.vectorize(random_key, vocab, vocab_size)
            random_key_vec = random_key_vec[0: sequence_length]
            random_key_vec = random_key_vec + (sequence_length - len(random_key_vec)) * [vocab_size]

            for values in random_values:        # values contains answer and label
                input_x1.append(random_key_vec)
                val_vec = DataProcessor.vectorize(values[0], vocab, vocab_size)
                val_vec = val_vec[0: sequence_length]
                val_vec = val_vec + (sequence_length - len(val_vec)) * [vocab_size]
                input_x2.append(val_vec)
                labels.append(values[1])
            batches.append([np.array(input_x1), np.array(input_x2), np.array(labels)])
        return batches

    def loadDataSets(self, vocab, vocab_size, file_path, sequence_length):
        # datasets contains dataSets[[question,answer],[],[],...] with string format
        dataSets = {}
        if not os.path.exists(file_path):
            print('--- file doesnot exist ---\n')
        file_reader = open(file_path,  encoding='utf-8', mode='r')
        for line in file_reader:
            items = line.strip().split('\t')
            q_standard = items[0]
            q_candidate = items[1]
            label = items[2]
            if dataSets=={}:
                dataSets[q_standard] = [[q_candidate, label]]  # {question:[[answer, label], ...]]
            elif (q_standard in dataSets.keys()):
                dataSets[q_standard].append([q_candidate, label])
            else:
                dataSets[q_standard] = [[q_candidate, label]] # {question:[[answer, label], ...]]
        return dataSets


if __name__ == '__main__':
    data_processor = DataProcessor()

    vocab_size = data_processor.get_vocabulary_size(FLAGS.vocab_path)
    vocab, revocab = DataProcessor.initialize_vocabulary(FLAGS.vocab_path)
    data_processor.get_init(FLAGS.input_training_data_path, FLAGS.input_validation_data_path, vocab, vocab_size,
                            FLAGS.max_length, revocab)

    rs  = data_processor.loadValData_step(vocab, vocab_size,FLAGS.input_validation_data_path, FLAGS.max_length)
    print('Done')