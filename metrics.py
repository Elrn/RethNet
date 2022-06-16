import tensorflow as tf
import numpy as np

########################################################################################################################
# Similarity Metric
########################################################################################################################

########################################################################################################################
""" Symmetrity """
########################################################################################################################
class _BASE(tf.keras.metrics.Metric):
    def __init__(self, num_classes, target=None, argmax=True, name=None, dtype=None, **kwargs):
        super(_BASE, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        normalize_tuple = lambda x:[x] if isinstance(x, int) else x
        self.target = normalize_tuple(target) if target != None else [i for i in range(1, num_classes)]
        self.non_target = [i for i in range(num_classes) if i not in self.target]
        if True in tf.math.greater_equal(self.target, num_classes):
            raise ValueError(
                f'Target label value must lower than num_class value, {target} < {num_classes}'
            )
        self.argmax = argmax
        self.CM = self.add_weight(
            'confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.compat.v1.zeros_initializer)


    def band_part_except_diag(self, CM, mode=0):
        if mode == 1:
            Upper = tf.linalg.band_part(CM, 0, -1)
            condition = tf.equal(Upper, 0)
        elif mode == -1:
            Lower = tf.linalg.band_part(CM, -1, 0)
            condition = tf.equal(Lower, 0)
        else:
            raise ValueError(f'mode value must be 1 or -1, but accepted {mode}')
        return tf.where(condition, CM, 0)

    def process_confusion_matrix(self, ):
        diag = tf.linalg.diag_part(self.CM)
        TP = tf.gather(diag, self.target)
        TN = tf.gather(diag, self.non_target)

        zeros_diag = tf.linalg.set_diag(self.CM, np.zeros([self.num_classes]))
        FP = [zeros_diag[:, i] for i in self.target]
        FN = [zeros_diag[:, i] for i in self.non_target]

        self.TP = tf.reduce_sum(TP)
        self.TN = tf.reduce_sum(TN)
        self.FP = tf.reduce_sum(FP)
        self.FN = tf.reduce_sum(FN)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype) # None 일 경우 float 할당
        y_pred = tf.cast(y_pred, self._dtype)
        if self.argmax:
            y_true = tf.argmax(y_true, -1)
            y_pred = tf.argmax(y_pred, -1)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        CM = tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]),
            tf.reshape(y_pred, [-1]),
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype)

        return self.CM.assign_add(CM)

    def result(self):
        self.process_confusion_matrix()

    def reset_state(self):
        tf.keras.backend.set_value(
            self.CM, np.zeros((self.num_classes, self.num_classes))
        )
    def get_config(self):
        config = super(_BASE, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })

        return config

########################################################################################################################
class DSC(_BASE):
    """ Dice Similarity Coefficients == F1 score """
    def __init__(self, num_classes, target=None, name='DSC', dtype=None, **kwargs):
        super(DSC, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return tf.math.divide_no_nan(
            2 * self.TP,
            2 * self.TP + self.FP + self.FN)

########################################################################################################################
class RVD(_BASE):
    """ relative volume difference """
    def __init__(self, num_classes, target=None, name='RVD', dtype=None, **kwargs):
        super(RVD, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return tf.math.divide_no_nan(
            self.FP - self.FN,
            self.TP + self.FN)

########################################################################################################################
class JSC(_BASE):
    """ Jaccard Similarity Coefficient """
    def __init__(self, num_classes, target=None, name='JSC', dtype=None, **kwargs):
        super(JSC, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return tf.math.divide_no_nan(
            self.TP,
            self.TP + self.FP + self.FN)

#
class Precision(_BASE):
    def __init__(self, num_classes, target=None, name='Precision', dtype=None, **kwargs):
        super(Precision, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return tf.math.divide_no_nan(
            self.TP,
            self.TP + self.FP)
#
class Recall(_BASE):
    def __init__(self, num_classes, target=None, name='Recall', dtype=None, **kwargs):
        super(Recall, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return tf.math.divide_no_nan(
            self.TP,
            self.TP + self.FN)


class F_Score(_BASE):
    def __init__(self, num_classes, target=None, beta=2, name='F_Score', dtype=None, **kwargs):
        name = f'F{beta}_Score' if beta is not None else name
        super(F_Score, self).__init__(num_classes=num_classes, target=target, name=name, dtype=dtype, **kwargs)
        self.beta = beta
    def result(self):
        super().result()
        beta = self.beta * self.beta
        return tf.math.divide_no_nan(
            (1+beta) * self.TP,
            (1+beta) * self.TP + beta * self.FN + self.FP)

########################################################################################################################
""" Differenece Metric """
########################################################################################################################
class VOE(JSC):
    """ Volume Overlap Error """
    def __init__(self, num_classes, name='VOE', dtype=None, **kwargs):
        super(VOE, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)

    def result(self):
        JSC = super().result()
        return 1 - JSC

########################################################################################################################
class SVD(DSC): # FP - FN / TP + FN
    """ Singular Value Decomposition """
    def __init__(self, num_classes, name='SVD', dtype=None, **kwargs):
        super(SVD, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)

    def result(self):
        DSC = super().result()
        return 1 - DSC

########################################################################################################################
""" Boundary Dice """
########################################################################################################################
from scipy import ndimage
class surface_distance(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name=None, dtype=None, **kwargs):
        super(surface_distance, self).__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes

    def get_contour(self, x, connectivity=1):
        conn = ndimage.generate_binary_structure(x.ndim, connectivity)
        except_contour = ndimage.binary_erosion(x, conn)
        contour = x ^ except_contour
        return contour

    def get_distance(self, contour, sampling=1):
        return ndimage.distance_transform_edt(~contour, sampling)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        self.labels  = tf.split(y_true, self.num_classes, -1)
        self.predictions = tf.split(y_pred, self.num_classes, -1)
        # return

    def result(self):
        tmp = zip(
            [self.get_distance(label) for label in self.labels],
            [self.get_contour(pred) for pred in self.predictions]
        )
        return np.concatenate([distance[contour] for distance, contour in tmp])

    def get_config(self):
        config = super(_BASE, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config

########################################################################################################################
class ASSD(surface_distance):
    """ Average Symmetric Surface Distance """
    def __init__(self, num_classes, name='ASSD', dtype=None, **kwargs):
        super(ASSD, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return

########################################################################################################################
class MSSD(surface_distance):
    """ Maximum Symmetric Surface Distance """
    def __init__(self, num_classes, name='MSSD', dtype=None, **kwargs):
        super(MSSD, self).__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)

    def result(self):
        super().result()
        return