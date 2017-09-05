# coding=utf-8

import tensorflow as tf
import numpy as np
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 512
decoder_hidden_units = encoder_hidden_units * 2

encoder_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# encoder_input_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_input_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_target')
decoder_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_input')

embedding = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, encoder_input)

decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)

encoder_cell_fw = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_cell_bw = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                              cell_bw=encoder_cell_bw,
                                                              inputs=encoder_inputs_embedded,
                                                              dtype=tf.float32,
                                                              time_major=True))

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c,
                                                    h=encoder_final_state_h)


decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_input))
print(encoder_max_time)
print(batch_size)

# decoder_length = encoder_input_length + 3

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, encoder_outputs,
                                                         initial_state=encoder_final_state, dtype=tf.float32)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets,
                      depth=vocab_size,
                      dtype=tf.float32),
    logits=decoder_logits)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_ = [[6], [3, 4], [9, 8, 7]]

batch_, batch_length_ = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32), max_sequence_length=4)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
                 feed_dict={
                            encoder_input: batch_,
                            decoder_input: din_,})
print('decoder predictions:\n' + str(pred_))



