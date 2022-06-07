import layers
import tensorflow as tf
import modules


def base(filters, div=4, kernel=3):
    concat_list = []
    dilation = [1+2*i for i in range(div)]
    div_channel = filters // div
    attn = layers.sep_bias(div)
    def main(x):
        features = [modules.conv(div_channel, 1)(attn(x, i)) for i in range(div)]
        for i, feature in enumerate(features):
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same')(feature)
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same', groups=div_channel)(x)
            if concat_list:
                x = tf.add(concat_list[-1], x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        x = modules.BN_ACT(x)
        return x
    return main
