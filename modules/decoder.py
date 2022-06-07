import layers
import tensorflow as tf
from tensorflow.keras.layers import *
import modules

def base(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = layers.sep_bias(div)

    def main(x, skip):
        features = [modules.conv(div_channel, 1)(attn(x, i))
                    for i in range(div)]
        for feature in features:
            x = modules.convTranspose(div_channel, kernel, strides=2,
                                     padding='same')(feature)
            for j in range(1, 4):
                x += modules.conv(div_channel, kernel, dilation_rate=j,
                                 padding='same', groups=div_channel)(x)
            concat_list.append(x)
        concat_list.append(skip)
        x = tf.concat(concat_list, -1)
        x = modules.BN_ACT(x)
        return x

    return main