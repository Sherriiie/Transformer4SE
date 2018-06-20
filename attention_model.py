import os
import math
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import numpy as np
from attention_datapro import DataProcessor

class AttentionModel():
    def __int__(self):
        return
    # def __int__(self, vocab_path, max_length, embedding_size, win_size, conv_size, dense_size, share_weight):
    # def __int__(self):
    #     self.vocab_path = vocab_path
    #     self.vocab_size = self.get_vocabulary_size()
    #     self.vocab = lookup_ops.index_table_from_file(vocab_path, default_value=self.vocab_size)
    #     self.max_length = max_length
    #     self.embedding_size = embedding_size
    #     self.win_size = win_size
    #     self.conv_size = conv_size
    #     self.dense_size = dense_size
    #     self.share_weight = share_weight
    #     self.use_one_hot_embedding = False
    #     if self.embedding_size <= 0:
    #         self.embedding_size = self.vocab_size + 1
    #         self.use_one_hot_embedding = True
    #     return


    def embed(self,
              inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=True,
              scope="embedding",
              reuse=tf.AUTO_REUSE):
        """
        dense embedding layer
        :parameter text_index: [B, S'], batch_size, length of query
        '''Embeds a given tensor.

        Args:
          inputs: A `Tensor` with type `int32` or `int64` containing the ids
             to be looked up in `lookup table`.
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            :return #[B, S, Dim]
            """

        with tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)
            if scale:
                outputs = outputs * (num_units ** 0.5)
        return outputs

    '''
    [B, S, D]
    '''
    #
    def multihead_attention(self,
                            query, key, value,
                            num_unit=None,
                            num_heads=8,
                            is_training=True,
                            dropout_rate=0,
                            mask_future=False,
                            scope='multihead_attention',
                            reuse=tf.AUTO_REUSE):
        '''
        multihead attention: softmax((Q*K')*sqrt)*T
        :param query: [B, S, E]
        :param key:
        :param value:
        :param num_unit: linear project unit
        :return:
        '''

        with tf.variable_scope(scope, reuse=reuse):
            if num_unit == None:
                num_unit = query.get_shape().as_list()[-1]

            # linear projection
            query_projection = tf.layers.dense(query, num_unit, activation=tf.nn.relu)      # **todo is activation needed here?**
            key_projection = tf.layers.dense(key, num_unit, activation=tf.nn.relu)
            value_projection = tf.layers.dense(value, num_unit, activation=tf.nn.relu)

            # multi-attetnion
            Q = tf.concat(tf.split(query_projection, num_heads, axis=2), 0)     # [hB, S, num_unit/h], to check **todo **
            # Q = tf.tile(query_projection, [1, 1, num_heads])            # [B, S, num_unit*h]
            K = tf.concat(tf.split(key_projection, num_heads, axis=2), 0)
            # K = tf.tile(key_projection, [1, 1, num_heads])
            V = tf.concat(tf.split(value_projection, num_heads, axis=2), 0)
            # V = tf.tile(value_projection, [1, 1, num_heads])
            # for q, k, v in enumerate([Q, K, V]):
            #     pass
            output_ = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            output_ = tf.div(output_, tf.sqrt(tf.cast(num_unit, tf.float32)))  # [B, S_query, S_key]
            # query mask

            # future blinding
            if mask_future:
                dialog_matrix = tf.ones_like(output_[0, :, :])      #[S_query, S_key]
                dialog_matrix = tf.contrib.linalg.LinearOperatorLowerTriangular(dialog_matrix).to_dense()
                output_ = tf.matmul(output_, dialog_matrix)

            output_softmax = tf.nn.softmax(output_)
            # dropout 
            output_softmax = tf.layers.dropout(output_softmax, dropout_rate, is_training)

            output = tf.matmul(output_softmax, V)      # [B, S_query, num_unit]
            output = tf.concat(tf.split(output, num_heads, 0), -1)      # [B, S_query, num_unit*h]  **no cut, no concate**
            #linear projection to the dim of num_unit
            # output = tf.layers.dense(output, num_unit)

            # Residual connection
            output = output + query
            return self.batch_normalize(output)         #Normalize, [B, S, num_unit]

    def feed_forward(self, inputs, units, scope='feed_forwad', reuse=tf.AUTO_REUSE):
        '''
        feed forward layer
        :param inputs: [B, S, num_unit]
        :param units:
        :param scope:
        :param reuse:
        :return: output [B, S, num_unit]
        '''
        with tf.variable_scope(scope,reuse=reuse):
            # layer 1
            output = tf.layers.conv1d(inputs, filters=units[0], kernel_size=1, activation=tf.nn.relu)
            # layer 2
            output = tf.layers.conv1d(output, filters=units[1], kernel_size=1, activation=tf.nn.relu)
            # residual connection
            output = output + inputs
            # normalize
            output = self.batch_normalize(output)
            return output





    def batch_normalize(self,
                  inputs,
                  epsilon=1e-8,
                  scope="batch_nr",
                  reuse=tf.AUTO_REUSE):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            beta = tf.Variable(tf.zeros_like(inputs[0,0,:]))
            gamma = tf.Variable(tf.ones_like(inputs[0,0,:]))
            mean, variance = tf.nn.moments(inputs, -1, keep_dims=True)
            normalized = tf.div((inputs - mean), tf.sqrt(variance+epsilon))
            outputs = tf.multiply(normalized, gamma) + beta
            # inputs_shape = inputs.get_shape()
            # params_shape = inputs_shape[-1:]
            # normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            # outputs = gamma * normalized + beta
        return outputs

    def vec_normalize(self,
                      inputs,
                      epsilon=1e-8,
                      scope="vector_nr",
                      reuse=tf.AUTO_REUSE):
        """
        apply normalization on vector to make vector length = 1
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            inputs_len = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), 1))
            inputs_len_wise = tf.tile(tf.expand_dims(inputs_len, -1), [1, inputs.get_shape()[1]])
            outputs = tf.div(inputs, inputs_len_wise)
        return outputs





#
# class Evaluate():
#
#     def inference_index(self, query_index, doc_index):
#         """
#         query and doc deep embedding inference for index input
#         """
#         qnvec = self.forward_propagation_index(query_index)
#         if self.share_weight:
#             dnvec = self.forward_propagation_index(doc_index)
#         else:
#             dnvec = self.forward_propagation_index(doc_index, 'd')
#         return qnvec, dnvec
#
#     def forward_propagation_index(self, text_index, model_prefix='q'):
#         """
#         forward propagation process for index input
#         """
#         text_embedding = self.embedding_layer(text_index, model_prefix)
#         maxpooling = self.conv_maxpooling_layer(text_embedding, model_prefix)
#         normalized_vec = self.dense_layer(maxpooling, model_prefix)
#         return normalized_vec








