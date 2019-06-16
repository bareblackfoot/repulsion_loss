import tensorflow as tf
from model.config import cfg

def instead(data):
    return data

def zerof():
    return tf.constant(0.0, dtype=tf.float32)

def zerof2():
    return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

def concat(a, b):
    return tf.concat([a, b], axis=0)