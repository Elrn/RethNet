import modules
import tensorflow as tf
from tensorflow.keras.layers import *
import layers
from modules import *

########################################################################################################################

########################################################################################################################
def base(n_class, base_filters=64, ):
    N = [40, 20, 10]
    # N = [32, 16, 8]
    # filters = [base_filters * i for i in range(1, 5)]
    filters = [64, 128, 256, 512]

    def main(x):
        x = layers.resize(0.5)(x)
        x = encoder.Double_Convolution(filters[0])(x)
        # encoding
        for i in range(2):
            x, x_ = encoder.base(filters[i], patch_size=N[i])(x)
            skip = modules.conv(48, 1, padding='same')(x_) if i == 0 else skip
        # latent
        for i in range(3):
            x, _ = encoder.base(filters[2], patch_size=N[2], pool=None)(x)
        x, _ = encoder.base(filters[-1], patch_size=N[-1], pool=None)(x)
        x = encoder.Xception(filters[-1])(x)
        # ASPP
        x = encoder.ASPP(filters[2])(x)
        x = modules.conv(256, 1, padding='same')(x)
        x = layers.resize(4)(x)
        x = tf.concat([x, skip], -1)
        x = encoder.Double_Convolution(filters[2], pool=None)(x)
        x = layers.resize(4)(x)
        # affine
        x = modules.conv(17, 1, padding='same')(x)
        x = modules.conv(n_class, 1, padding='same')(x)
        output = Softmax(-1)(x)
        return output
    return main
