import layers
import tensorflow as tf

import modules

def REthinker(filters, N=None):
    [1,  1]
    def main(x):
        skip = layers.SE(0.125)(x)

        x = tf.image.extract_patches(x, )
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
