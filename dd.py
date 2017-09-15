import tensorflow as tf
import numpy as np
logits = [(1,2,3,4,5,6,7,8,9)]
# logits = [[[1,2,3], [2,3,4]],[[4,5,6],[5,6,7]],[[7,8,9],[8,9,6]]]
# print(np.shape(logits))
# logits = np.reshape(logits, [-1])
# a = tf.argmax(logits, axis=-1)
print np.shape(logits)
print logits