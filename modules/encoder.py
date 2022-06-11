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
        def reconstruction(patch_size):
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
                shape = [-1, h * patch_size, c // patch_size, w]
                x = tf.reshape(x, shape)
                x = tf.transpose(x, [0,1,3,2])
                b, h, w, c = x.shape
                shape = [-1, h, w * patch_size, c // patch_size]
                x = tf.reshape(x, shape)
                return x
            return main

        def main(x):
            _, psz, x_nP, y_nP, n_ch= x.shape
            x = tf.transpose(x, [0, 2, 3, 1, 4]) # [_, nP_x, nP_y, x_psz * y_psz, C]
            x = tf.reshape(x, [-1, x_nP, y_nP, n_ch*psz])
            x = reconstruction(patch_size)(x)
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

        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

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
        x *= skip
        return x
    return main

def Xception(filters):
    def main(x):
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = Dense(filters)(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        return x
    return main

def base(filters, patch_size, pool=2):
    def main(x):
        x = Xception(filters)(x)
        x = REthinker(filters, patch_size)(x)
        skip = x
        if pool != None:
            x = modules.pooling(pool, pool)(x)
        return x, skip
    return main

def ASPP(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = layers.sep_bias(div)

    def main(x):
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        features = [modules.conv(div_channel, 1)(attn(x, i))
                    for i in range(div)]
        for feature in features:
            x = modules.conv(div_channel, kernel,
                                      padding='same')(feature)
            for j in range(1, 4):
                x += modules.conv(div_channel, kernel, dilation_rate=j,
                                  padding='same', groups=div_channel)(x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        return x
    return main

def Double_Convolution(filters, pool=2):
    def main(x):
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        if pool != None:
            x = modules.pooling(pool, pool)(x)
        return x
    return main
