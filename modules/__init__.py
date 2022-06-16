__all__ = [
    'encoder',
]

from tensorflow.keras.layers import *
import layers
import tensorflow as tf

rank = None

conv = None
convTranspose = None
pooling = None
depthwise = None

def assignment_function_according_to_data_rank():
    global conv
    global convTranspose
    global pooling
    global depthwise

    if rank == 2:
        conv = Conv2D
        convTranspose = Conv2DTranspose
        pooling = AveragePooling2D
        depthwise = DepthwiseConv2D
    elif rank == 3:
        conv = Conv3D
        convTranspose = Conv3DTranspose
        pooling = AveragePooling3D
        depthwise = layers.DepthwiseConv3D
    else:
        raise ValueError(f'D is must 2 or 3, not "{rank}".')




BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
