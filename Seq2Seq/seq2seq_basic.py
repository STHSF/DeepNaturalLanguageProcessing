import numpy as np
import tensorflow as tf
from Seq2Seq import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 25
decoder_hidden_units = encoder_hidden_units

# encoder_inputs:[max_time, batch_size]
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# decoder_targets: [max_time, batch_size]
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
# decoder_inputs: [max_time, batch_size]
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0),
                         dtype=tf.float32)
# encoder_inputs_embeded: [max_time, batch_size, input_embedding_size]
encoder_inputs_embeded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# decoder_inputs_embeded: [max_time, batch_size, input_embedding_size]
decoder_inputs_embeded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                         encoder_inputs_embeded,
                                                         dtype=tf.float32, time_major=True)
del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                         decoder_inputs_embeded,
                                                         initial_state=encoder_final_state,
                                                         dtype=tf.float32, time_major=True,
                                                         scope='plain_decoder')

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
                 feed_dict={encoder_inputs: batch_,
                            decoder_inputs: din_,})
print('decoder predictions:\n' + str(pred_))