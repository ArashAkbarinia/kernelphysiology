'''
Collection of metrics and losses.
'''


from keras import backend as K
import tensorflow as tf


def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def reproduction_angular_error():
    def error(y_true, y_pred):
        y_pred_sum = K.tile(K.sum(y_pred, axis=1, keepdims=True), [1, 3])
        y_true_sum = K.tile(K.sum(y_true, axis=1, keepdims=True), [1, 3])
        y_pred = y_pred / K.reshape(y_pred_sum, (-1, 3))
        y_true = y_true / K.reshape(y_true_sum, (-1, 3))

        l2l1 = y_true / y_pred
        l2l1_sum = K.tile(K.sum(l2l1 ** 2, axis=1, keepdims=True), [1, 3])
        w1 = l2l1 / K.reshape(l2l1_sum ** 0.5, (-1, 3))
        w2 = 1.0 / (3.0 ** 0.5)

        w1w2 = K.sum(w1 * w2, axis=1)
        w1w2 = K.minimum(w1w2, 1)
        w1w2 = K.maximum(w1w2, -1)
        r = tf.math.acos(w1w2)
        r = tf_rad2deg(r)
        return K.mean(r, axis=-1)
    return error


def mean_absolute_error():
    def loss(y_true, y_pred):
        y_pred_sum = K.tile(K.sum(y_pred, axis=1), 3)
        y_true_sum = K.tile(K.sum(y_true, axis=1), 3)
        y_pred = y_pred / K.reshape(y_pred_sum, (-1, 3))
        y_true = y_true / K.reshape(y_true_sum, (-1, 3))
        return K.mean(K.abs(y_pred - y_true), axis=-1)
    return loss