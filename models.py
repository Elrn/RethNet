import modules
import tensorflow as tf
from tensorflow.keras.layers import *
import layers
from modules import *

########################################################################################################################

########################################################################################################################
def base(n_class, base_filters=64, ):
    N = [32, 16, 8]
    # filters = [base_filters * i for i in range(1, 5)]
    filters = [64, 128, 256, 512]

    def main(x):
        layers.resize(0.5)(x)
        x = Conv2D(filters[0], padding='same')(x)
        x = Conv2D(filters[0], padding='same')(x)
        # encoding
        for i in range(0, 2):
            x = encoder.base(filters[i], N=N[i])(x)
            skip = Conv2D(48, 1, padding='same')(x) if i == 0 else skip
        # latent
        for i in range(3):
            x = encoder.base(filters[2], N=N[2])(x)
        x = encoder.base(filters[-1], N=N[-1])(x)
        x = encoder.Xception(filters[-1])(x)
        # ASPP
        x = encoder.ASPP(filters[2])
        x = modules.conv(256, 1, padding='same')(x)
        x = layers.resize(4)(x)
        x = tf.concat([x, skip], -1)
        x = modules.conv(256, 3, padding='same')(x)
        x = modules.conv(256, 3, padding='same')(x)

        x = layers.resize(4)(x)
        x = modules.conv(17, 1, padding='same')(x)

        x = modules.conv(n_class, 1, padding='same')(x)
        output = Softmax(-1)(x)
        return output
    return main
