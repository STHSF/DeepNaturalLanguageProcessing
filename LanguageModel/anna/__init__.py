#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: __init__.py.py
@time: 2017/7/3 上午9:56
"""

import numpy as np
import tensorflow as tf

# sess = tf.Session()
#
# t = tf.constant([[1.,1.,1.], [3.,3.,3.]])
# t1 = tf.constant([[True, True], [False, False]])
#
# print(sess.run(tf.reduce_any(t1, 1, keep_dims=True)))
#
# sess.close()

a = np.random.choice(5, 1, p=[0.2, 0.8, 0])

print(a.shape)