__all__ = [
    'decoder',
    'encoder',
    'latent',
    'skip_connection',
]

from tensorflow.keras.layers import *
import layers
import tensorflow as tf


act = 'relu'
D = 2
if D == 2:
    conv = Conv2D
    convTranspose = Conv2DTranspose
    pooling = AveragePooling2D
    depthwise = DepthwiseConv2D
elif D == 3:
    conv = Conv3D
    convTranspose = Conv3DTranspose
    pooling = AveragePooling3D
    depthwise = layers.DepthwiseConv3D
else:
    raise ValueError(f'D is must 2 or 3, not "{D}".')

BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
ACT_BN = lambda x : BatchNormalization()(tf.nn.relu(x))