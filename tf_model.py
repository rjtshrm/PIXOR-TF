import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, LeakyReLU, Add

def conv3x3(out_channels, stride=1, bias=False):
    return Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding='same', use_bias=bias)


def conv1x1(out_channels, stride=1, bias=False):
    return Conv2D(filters=out_channels, kernel_size=1, strides=stride, use_bias=bias)


def bottelneck(input, channels, block_expansion, stride=1, downsample=None, use_bn=True):

    bias = not use_bn
    identity = input

    x = conv1x1(channels, stride=1, bias=bias)(input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.0)(x)

    x = conv3x3(channels, stride=stride, bias=bias)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.0)(x)

    x = conv1x1(channels * block_expansion, stride=1, bias=bias)(x)
    if use_bn:
        x = BatchNormalization()(x)

    if downsample is not None:
        identity = downsample(input)

    x += identity
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.0)(x)

    return x

def _make_layer(inputs, channels, num_blocks, use_bn):

    block_expansion = 4

    if use_bn:
        downsample = Sequential([
            Conv2D(channels * block_expansion, kernel_size=1, strides=2, use_bias=False),
            BatchNormalization()
            ]
        )
    else:
        downsample = Conv2D(channels * block_expansion,
                               kernel_size=1, strides=2, use_bias=True)


    x = bottelneck(inputs, channels, block_expansion=block_expansion, stride=2, downsample=downsample, use_bn=use_bn)

    for i in range(1, num_blocks):
        x = bottelneck(x, channels, block_expansion=block_expansion, stride=1, use_bn=use_bn)
    return x

def head(input, use_bn, use_height):
    bias = not use_bn

    x = input

    for i in range(4):
        x = Conv2D(filters=96, kernel_size=3, padding='same', strides=1, use_bias=bias)(x)
        if use_bn:
            x = BatchNormalization()(x)


    cls = Activation('sigmoid')(Conv2D(filters=1, kernel_size=3, padding='same', strides=1, use_bias=True)(x))

    if use_height:
        regression_channels = 8
    else:
        regression_channels = 6
    loc = Conv2D(filters=regression_channels, kernel_size=3, padding='same', strides=1)(x)

    op = tf.concat((cls, loc), axis=-1)

    return op


def pixor_modified(input_shape=(800, 700, 36), num_block=[3, 6, 6, 3], use_bn=True, use_height=False):
    '''
        A nn model based on Pixor
        refer: https://arxiv.org/pdf/1902.06326.pdf
    '''

    input = Input(shape=input_shape)

    bias = not use_bn

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=bias)(input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.0)(x)

    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=bias)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.0)(x)

    # res-1
    b1 = _make_layer(inputs=x, channels=24, num_blocks=num_block[0], use_bn=use_bn)

    # res-2
    b2 = _make_layer(inputs=b1, channels=48, num_blocks=num_block[1], use_bn=use_bn)

    # res-3
    b3 = _make_layer(inputs=b2, channels=64, num_blocks=num_block[2], use_bn=use_bn)

    # res-4
    b4 = _make_layer(inputs=b3, channels=96, num_blocks=num_block[3], use_bn=use_bn)

    # lateral layers
    l5 = Conv2D(filters=196, kernel_size=1, activation='linear', strides=1, padding='same')(b4)

    l4 = Conv2D(filters=128, kernel_size=1, activation='linear', strides=1, padding='same')(b3)
    l4_deoconv = Conv2DTranspose(filters=128, kernel_size=3, activation='linear', strides=2, padding='same')(l5)
    #l4 = Add()([l4, l4_deoconv])
    l4 += l4_deoconv

    l3 = Conv2D(filters=96, kernel_size=1, activation='linear', strides=1, padding='same')(b2)
    l3_deoconv = Conv2DTranspose(filters=96, kernel_size=3, activation='linear', strides=2, padding='same', output_padding=(1, 0))(l4)
    #l3 = Add()([l3, l3_deoconv])
    l3 += l3_deoconv

    cls_reg = head(l3, use_bn=use_bn, use_height=use_height)

    model = Model(inputs=input, outputs=cls_reg)

    return model


if __name__ == '__main__':
    m = pixor_modified()
    #print(m.summary())
    a = np.zeros((5, 800, 700, 36))
    a = tf.convert_to_tensor(a)
    op = m.predict(a, 5)
    print(op.shape)

