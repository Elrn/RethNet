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
        x = Conv2D(filters[0])(x)
        x = Conv2D(filters[0])(x)
        # encoding
        for i in range(0, 2):
            x = encoder.base(filters[i], N=N[i])(x)
        # latent
        for i in range(3):
            x = encoder.base(filters[2], N=N[2])(x)
        x = encoder.base(filters[-1], N=N[-1])(x)
        x = encoder.Xception(filters[-1])(x)
        # decoding
        for i in reversed(range(depth)):
            x = decoder.base(filters[i])(x)
        x = modules.conv(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main
