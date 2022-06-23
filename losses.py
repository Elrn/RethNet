import logging

from tensorflow.python.keras.losses import *
from keras.utils import losses_utils
import tensorflow as tf
from tensorflow import reduce_mean as E

import numpy as np
from scipy import ndimage

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
# from keras.backend import categorical_crossentropy

def get_mask(y_pred):
    """ return mask(0, 1) corresponding to prediction """
    n_ch = y_pred.shape[-1]
    y_pred = tf.math.argmax(y_pred, -1)
    y_pred = tf.one_hot(y_pred, n_ch)
    return y_pred

########################################################################################################################
def cross_entropy(y_true, y_pred, gamma=1.7):
    """
    :param gamma: 반비례
    :return: [B, (I), C]
    """
    prediction_mask = get_mask(y_pred)
    condition = tf.equal(y_true - prediction_mask, -1)  # FP
    prediction = tf.where(condition, 1 - y_pred, y_pred) # TP, FP
    prediction = tf.clip_by_value(prediction, EPSILON, 1. - EPSILON)
    loss = -tf.math.log(prediction)
    loss *= prediction_mask  # only part of prediction
    loss = loss ** gamma
    return loss # B, H, W, C


def CE(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred)
    tf.reduce_mean(loss)
    return loss  # B, H, W, C

########################################################################################################################
def WCE(distance_rate=0.04):
    """
    Positive Loss의 경우 background 를 넓혀 loss를 줄이려는 경향을 보임
    FN Loss의 경우 FP 부분을 pred_count에 반영할 것인지 여부 확인
    """
    def get_distance_weight_map(y_true):
        """
        patch의 경우 상대적 distance 추가 필요?
        :return: [B, (I), C]
        """
        @tf.function(input_signature=[tf.TensorSpec(y_true.shape, tf.float32)])
        def tf_fn(x):
            x = 1. + tf.numpy_function(
                ndimage.distance_transform_edt, [x, distance_rate], np.double
            )
            return x
        return tf.cast(tf_fn(y_true), tf.float32)

    def frequency_ratio_by_class(y_true):
        """
        patch 혹은 slice 의 경우 일부 label 이 없을 수 있다. 따라서 수동으로 기입해주는 것이 좋다.
        :return [C]
        """
        rank = len(y_true.shape)
        total = np.prod(y_true.shape[:-1]) # BatchNormalization
        axis = [i for i in range(rank - 1)] # until channel axis
        amount_by_class = tf.reduce_sum(y_true, axis)
        ratio = amount_by_class / total
        return ratio

    def freq_weight_map(y_true, y_pred):
        """
        relative small object로 predinction 하는 weight은 조금 더 낮은 값을 할당
        :return: [B, (I), C]
        """
        def weight_fn(x):
            # x = -tf.math.log(0.96 * x + 0.03)
            # x = x ** 0.55 + 0.17
            condition = tf.math.greater(x, 0.) # as small object

            x = tf.math.abs(x)
            weight = tf.where(condition, tf.math.exp(x ** 1.3), tf.math.exp(x ** 1.7))
            # lower loss for TP
            condition = tf.equal(weight, 1.)
            weight = tf.where(condition, 0.9, weight)
            return weight

        ratio = frequency_ratio_by_class(y_true)
        y = tf.math.argmax(y_true, -1) # int type
        y_ = tf.math.argmax(y_pred, -1) # int type
        condition = tf.equal(y, y_)
        # cast to 'float' to operate tf.where function
        y = tf.where(condition, 0., tf.cast(y, tf.float32))
        y_ = tf.where(condition, 0., tf.cast(y_, tf.float32))
        # cast to 'int' for index operation
        y = tf.gather(ratio, tf.cast(y, tf.int32))
        y_ = tf.gather(ratio, tf.cast(y_, tf.int32))
        diff = y - y_
        freq_weight = weight_fn(diff)
        freq_weight = tf.expand_dims(freq_weight, -1)
        return freq_weight

    def get_scale_factor(y_true, y_pred, gamma=0.3):
        """
        small object의 모든 pixel을 FP로 산정한 값이기 때문에 gamma 값을 1보다 높이는 것이 옳다.
        :param gamma: 높을 수록 작은 object에 대한 prediction loss 가 높아진다.
        :return [B, (I), C]
        """
        ratio = frequency_ratio_by_class(y_true)
        condition = tf.equal(ratio, 0.)
        ratio = tf.where(condition, 1., ratio)

        factor = 1 / ((ratio+0.001) ** gamma) - 0.2 # Prevent Nan loss value, +0.001
        factor_map = tf.reduce_max(y_true * factor, -1) # B (I)
        # for apply only FP
        condition = tf.equal(y_true - get_mask(y_pred), -1)
        FP = tf.where(condition, 1., 0.) # [B (I) C]
        scaled_map = FP * tf.expand_dims(factor_map, -1)

        # zero to one for TP
        condition = tf.equal(scaled_map, 0.)
        scaled_map = tf.where(condition, 1., scaled_map)
        return scaled_map

    def existance_CE(y_true, y_pred):
        """
        잘못된 predictiion 에 대해 penalty
        Background 같은 label의 FN에 penalty여부 효과 확인 요망
        """
        def weight_fn(x, gamma=2.0):
            x = (x - 1.) ** 2
            return gamma * x + 1

        axis = [i for i in range(tf.rank(y_true)-1)]  # until channel axis
        # label = tf.reshape(tf.argmax(y_true, -1), [-1])
        # pred = tf.reshape(tf.argmax(y_pred, -1), [-1])
        # CM = tf.math.confusion_matrix(label, pred)
        # pred_count_by_class = tf.cast(tf.reduce_sum(CM, 0), tf.float32)
        TP = y_true * get_mask(y_pred)
        TP_count_by_class = tf.reduce_sum(TP, axis)
        label_count_by_class = tf.reduce_sum(y_true, axis)
        # if label count 0 then get max weight
        ratio = weight_fn(tf.math.divide_no_nan(TP_count_by_class, label_count_by_class)) # 1 +- ratio by class
        FN = tf.where(tf.equal(y_true - get_mask(y_pred), 1), 1., 0.)
        FN = tf.clip_by_value(FN * y_pred, EPSILON, 1. - EPSILON)
        return ratio * -tf.math.log(FN)

    slice_ = lambda x, idx : tf.split(x, x.shape[-1], axis=-1)[idx]
    def label_relation(y_true, y_pred, condition=None):
        pred_map = get_mask(y_pred)
        condition = [[4, 5, 0.5]]

        slice_(y_true, 4) * slice_(pred_map, 5) * scale

        condition = tf.equal(y_true - get_mask(y_pred), -1)
        FP = tf.where(condition, 1., 0.) # [B (I) C]
        return

    def main(y_true, y_pred):
        loss = cross_entropy(y_true, y_pred)
        ### Weighting
        # loss *= get_distance_weight_map(y_true)
        loss *= get_scale_factor(y_true, y_pred)
        loss *= freq_weight_map(y_true, y_pred)
        return tf.reduce_sum(loss)
    return main

########################################################################################################################
def EL(y_true, y_pred, frequences=None, gamma=0.3): # B, H, W, C
    """
    3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
        https://arxiv.org/abs/1809.00076

    :param frequences: Imbalancing에 대한 class별 weight 값
    """
    def get_frequences_by_class(x):
        """
        patches 혹은 slices로 학습을 진행할 때
        병변 부위가 없는 batch의 경우 imbalance label overfitting 우려 존재
        따라서 수동 기입을 권장
        """
        weight_fn = lambda x: (-np.log(x))**0.5

        total = np.prod(x.shape[:-1])
        amount_by_class = tf.reduce_sum(x, [i for i in range(len(x.shape)-1)])

        return weight_fn(amount_by_class / total)

    if frequences == None:
        frequences = get_frequences_by_class(y_true)
        assert len(frequences) == y_true.shape[-1]
    logging.warning(f"[Loss] EL loss's gamma recommended '0 ~ 1', but '{gamma}'.")
    axis = [i for i in range(len(y_true.shape) - 1)] # except Batch, Channel

    def get_DICE(gamma=0.3):
        TP = tf.where(tf.math.equal(y_true, get_mask(y_pred)), 1, 0)
        numerator = 2 * tf.reduce_sum(TP * y_pred, axis) + EPSILON
        denominator = tf.reduce_sum(TP + y_pred, axis) + EPSILON

        DICE = -tf.math.log(numerator / denominator)
        DICE = DICE ** gamma
        return E(DICE, -1)

    DICE = get_DICE()
    CE = cross_entropy(y_true, y_pred)
    CE = (frequences * CE) ** gamma
    return DICE + CE

########################################################################################################################
def focal(y_true, y_pred, gamma=0.3):
    """
    Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
    """
    CE = cross_entropy(y_true, y_pred)
    return CE * (1 - y_true * y_pred) ** gamma

########################################################################################################################
# class custom_loss_template(LossFunctionWrapper):
#     def __init__(self,
#                  from_logits=False,
#                  label_smoothing=0.,
#                  axis=-1,
#                  reduction=losses_utils.ReductionV2.AUTO,
#                  name='custom_loss_template'
#                  ):
#         super().__init__(
#             self.SR,
#             name=name,
#             reduction=reduction,
#             from_logits=from_logits,
#             label_smoothing=label_smoothing,
#             axis=axis
#         )
#
#     def SR(self, y_true, y_pred, from_logits=False, label_smoothing=0., axis=-1, tau=2):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
#
#         y_true = tf.nn.softmax(y_true) ** (1 / tau)
#         y_pred = tf.nn.softmax(y_pred) ** (1 / tau)
#
#         def _smooth_labels():
#             num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
#             return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
#
#         y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing, _smooth_labels,
#                                                        lambda: y_true)
#
#         return tau**2 * backend.categorical_crossentropy(
#             y_true, y_pred, from_logits=from_logits, axis=axis)

########################################################################################################################
