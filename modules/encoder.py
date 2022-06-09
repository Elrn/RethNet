import layers
import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

import modules



#######################################################################################################################
def REthinker(filters, patch_size=None):
    sizes = [1, patch_size, patch_size, 1]

    def pre_processing(C):
        """
        [_, n_P_x, n_P_y, C * x_psz * y_psz]
        [_, x_psz*y_psz, n_P_x, n_P_y, C]

        Args:
            C: 패치 추출 이전의 원형 채널

        """
        def sort_by_channel(x):
            # Need to sort by channel since tf.extract_patches give channel mixed output
            _, x_psz, y_psz, pcp = x.shape
            x = tf.reshape(x, [-1, x_psz, y_psz, pcp//C, C])
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            x = tf.reshape(x, [-1, x_psz, y_psz, pcp])
            return x

        def main(x):
            _, xpsz, ypsz, ppc = x.shape
            x = sort_by_channel(x)
            x = tf.reshape(x, [-1, xpsz, ypsz, C, ppc//C])
            x = tf.transpose(x, [0, 4, 1, 2, 3])
            return x
        return main

    def post_processing():
        def reconstruction(N):
            """
            args:
                N: Patch size

            inputs:
                x: extracted patches that rank must be 4

            return:
                reconstructed image
            """
            def main(x):
                assert len(x.shape) == 4

                x = tf.transpose(x, [0,1,3,2])
                b, h, c, w = x.shape
                shape = [-1, h*N, c//N, w]
                x = tf.reshape(x, shape)
                x = tf.transpose(x, [0,1,3,2])
                b, h, w, c = x.shape
                shape = [-1, h, w*N, c//N]
                x = tf.reshape(x, shape)
                return x
            return main

        def main(x):
            reconstruction()
            return x
        return main

    def encoding(x, mode=1):
        """
        Conv3D / ConvLSTM

        `x.shape` 에 관한 문제,
        extract patches 의 output은 channel이 섞인 patch가 channel axis로 출력된다.
        따라서 ConvLSTM의 input에 맞도록 변환해주어야 한다.

        """
        if mode == 0:
            x = Conv3D(filters, kernel=3, padding='same',
                       # activation='tanh'
                       )(x)
            x = BatchNormalization()(x)
            x = tf.math.tanh(x)
        else:
            x = ConvLSTM2D(filters, 1, return_sequences=True)(x) # activation='tanh'
        return x

    def main(x):
        _, H, W, C = x.shape
        assert H % patch_size == 0 and W % patch_size == 0
        skip = layers.SE(0.125)(x)

        x = tf.image.extract_patches(
            images=x,
            sizes=sizes,
            strides=sizes,
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        x = pre_processing(C)(x)
        x = encoding(x, mode=1)
        x = post_processing()(x)
        x += skip
        return x
    return main

def Xception(filters):
    def main(x):
        x = Dense(filters)(x)
        x = modules.conv(filters, kernel=3, padding='same', groups=filters)(x)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        return x
    return main

def base(filters, pool=2):
    def main(x):
        x = Xception(filters)(x)
        x = REthinker(filters)(x)
        if pool != None:
            x = modules.pooling(pool, pool)(x)
        return x
    return main
