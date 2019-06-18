import tensorflow as tf


def IoG(box_a, box_b):
    inter_xmin = tf.maximum(box_a[:, 0], box_b[:, 0])
    inter_ymin = tf.maximum(box_a[:, 1], box_b[:, 1])
    inter_xmax = tf.minimum(box_a[:, 2], box_b[:, 2])
    inter_ymax = tf.minimum(box_a[:, 3], box_b[:, 3])
    Iw = tf.maximum(inter_xmax - inter_xmin + 1, 0.0)
    Ih = tf.maximum(inter_ymax - inter_ymin + 1, 0.0)
    I = Iw * Ih
    G = (box_a[:, 2] - box_a[:, 0] + 1) * (box_a[:, 3] - box_a[:, 1] + 1)
    return I / G, I