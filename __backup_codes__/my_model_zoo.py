import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, DenseNet121, DenseNet169, DenseNet201, VGG16
from tensorflow.keras import backend, regularizers, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose, Add, Concatenate, \
    MaxPooling2D, Input, AveragePooling2D, UpSampling2D, LeakyReLU, ReLU, Dropout, Cropping2D, SpatialDropout2D, \
    Multiply, \
    Conv3D, Conv3DTranspose, MaxPooling3D

from types import MethodType
import os
from tensorflow.python.keras.utils import data_utils
import numpy as np


class HRNet:
    def __init__(self, input_shape, n_class, nblocks=4, n_layer_per_block=3, n_filters=32, expansion=1.5,
                 use_plain_cove=True):
        self.input_shape = input_shape
        self.n_class = n_class
        self.nblocks = nblocks
        self.n_layer_per_block = n_layer_per_block
        self.n_filters = n_filters
        self.expansion = expansion
        self.use_plain_cove = use_plain_cove

        self.model_notop = self._build_hrnet_notop()
        self.model = self._build_harnet()

    def _res_block(self, input, n_fs):
        input = Conv2D(filters=n_fs, kernel_size=3, padding='same', kernel_initializer='he_normal')(input)
        input = BatchNormalization()(input)
        input = Activation(activation='relu')(input)

        x = Conv2D(filters=n_fs, kernel_size=3, padding='same', kernel_initializer='he_normal')(input)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        x = Conv2D(filters=n_fs, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Add()([input, x])
        out = Activation(activation='relu')(x)
        return out

    def _basic_block(self, input_tensor, n_filters, n_layer, plain_layer=None):
        x = input_tensor
        for _ in range(n_layer):
            if plain_layer:
                x = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
                x = BatchNormalization()(x)
                x = Activation(activation='relu')(x)
            else:
                x = self._res_block(x, n_fs=n_filters)
        return x

    def _bottleneck(self, inplanes, nfilters, downsample=False):
        later_planes = list()
        n_planes = len(inplanes)
        for i, plane in enumerate(inplanes):
            bk_ = Conv2D(filters=nfilters, kernel_size=3, padding='same', kernel_initializer='he_normal')(plane)
            later_planes.append(bk_)
            if n_planes - 1 == i and downsample:
                down_sp = Conv2D(filters=nfilters, kernel_size=3, strides=2, padding='same',
                                 kernel_initializer='he_normal')(plane)
                down_sp = BatchNormalization()(down_sp)
                down_sp = Activation('relu')(down_sp)
                later_planes.append(down_sp)
        if n_planes > 1:
            for i, plane in enumerate(inplanes):
                for j, lplane in enumerate(later_planes):
                    if i < j and i != n_planes - 1:
                        tp = MaxPooling2D(pool_size=2 ** (j - i))(plane)
                        later_planes[j] = Add()([later_planes[j], tp])
                    else:
                        if i > j:
                            tp = UpSampling2D(size=2 ** (i - j))(plane)
                            later_planes[j] = Add()([later_planes[j], tp])
        return later_planes

    def _head_block_sseg(self, inplanes):
        x = inplanes[0]
        for i, plane in enumerate(inplanes, 0):
            if i > 0:
                x = Concatenate()([UpSampling2D(size=2 ** i)(plane), x])
        return x

    def _build_hrnet_notop(self):
        input = Input(self.input_shape)
        x = (input,)
        for i in range(self.nblocks):
            downsampling = True if i <= self.nblocks - 1 else False
            filters = np.int32(self.n_filters * (1 + i * self.expansion))
            outp = []
            for j, ix in enumerate(x, 1):
                with tf.name_scope('Route{}-{}'.format(i + 1, j)):
                    y = self._basic_block(ix, n_filters=filters, n_layer=self.n_layer_per_block,
                                          plain_layer=self.use_plain_cove)
                    outp.append(y)
            with tf.name_scope('Bottleneck{}'.format(i + 1)):
                x = self._bottleneck(outp, nfilters=filters, downsample=downsampling)
        output_notop = self._head_block_sseg(x)

        model = Model(inputs=input, outputs=output_notop)
        return model

    def _build_harnet(self):
        x = Input(self.input_shape)
        y = self.model_notop(x)
        output = Conv2D(filters=self.n_class, kernel_size=3, padding='same')(y)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        return Model(x, output)


class MEDNet:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

        self.model_notop = self._build_mednet_notop()
        self.model = self._build_mednet()

    def conv2d_bn(self, x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  dilation_rate=(1, 1),
                  name=None):
        """Utility function to apply conv + BN.

        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            dilation_rate=dilation_rate,
            name=conv_name)(x)
        x = BatchNormalization(name=bn_name)(x)
        x = ReLU()(x)
        return x

    def _build_mednet_notop(self):

        with tf.name_scope('input'):
            input_img = Input(shape=self.input_shape)
            x1 = self.conv2d_bn(input_img, 128, 3, 3)
        with tf.name_scope('block1'):
            b1 = self.conv2d_bn(x1, 64, 3, 3)
            b2 = self.conv2d_bn(x1, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(x1, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(x1, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            block1 = Concatenate(axis=3)([b1, b2, b3, b4])

        with tf.name_scope('block2'):
            x2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block1)
            x2 = ReLU()(x2)
            b1 = self.conv2d_bn(x2, 64, 3, 3)
            b2 = self.conv2d_bn(x2, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(x2, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(x2, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            block2 = Concatenate(axis=3)([b1, b2, b3, b4])

        with tf.name_scope('block3'):
            x3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block2)
            x3 = ReLU()(x3)
            b1 = self.conv2d_bn(x3, 64, 3, 3)
            b2 = self.conv2d_bn(x3, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(x3, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(x3, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            block3 = Concatenate(axis=3)([b1, b2, b3, b4])

        with tf.name_scope('block4'):
            x4 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block3)
            x4 = ReLU()(x4)
            b1 = self.conv2d_bn(x4, 64, 3, 3)
            b2 = self.conv2d_bn(x4, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(x4, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(x4, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            block4 = Concatenate(axis=3)([b1, b2, b3, b4])

        with tf.name_scope('mid'):
            xm = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block4)
            xm = ReLU()(xm)
            b1 = self.conv2d_bn(xm, 64, 3, 3)
            b2 = self.conv2d_bn(xm, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(xm, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(xm, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            mid = Concatenate(axis=3)([b1, b2, b3, b4])
            mid = Conv2DTranspose(64, 2, strides=2)(mid)

        with tf.name_scope('ublock4'):
            xu4 = Concatenate(axis=3)([mid, block4])
            b1 = self.conv2d_bn(xu4, 64, 3, 3)
            b2 = self.conv2d_bn(xu4, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(xu4, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(xu4, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            ublock4 = Concatenate(axis=3)([b1, b2, b3, b4])
            ublock4 = Conv2DTranspose(64, 2, strides=2)(ublock4)

        with tf.name_scope('ublock3'):
            xu3 = Concatenate(axis=3)([ublock4, block3])
            b1 = self.conv2d_bn(xu3, 64, 3, 3)
            b2 = self.conv2d_bn(xu3, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(xu3, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(xu3, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            ublock3 = Concatenate(axis=3)([b1, b2, b3, b4])
            ublock3 = Conv2DTranspose(64, 2, strides=2)(ublock3)

        with tf.name_scope('ublock2'):
            xu2 = Concatenate(axis=3)([ublock3, block2])
            b1 = self.conv2d_bn(xu2, 64, 3, 3)
            b2 = self.conv2d_bn(xu2, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(xu2, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(xu2, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            ublock2 = Concatenate(axis=3)([b1, b2, b3, b4])
            ublock2 = Conv2DTranspose(64, 2, strides=2)(ublock2)

        with tf.name_scope('ublock1'):
            xu1 = Concatenate(axis=3)([ublock2, block1])
            b1 = self.conv2d_bn(xu1, 64, 3, 3)
            b2 = self.conv2d_bn(xu1, 48, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b2 = self.conv2d_bn(b2, 64, 3, 3)
            b3 = self.conv2d_bn(xu1, 48, 1, 1)
            b3 = self.conv2d_bn(b3, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(xu1, 48, 1, 1)
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(2, 2))
            b4 = self.conv2d_bn(b4, 64, 3, 3, dilation_rate=(3, 3))
            ublock1 = Concatenate(axis=3)([b1, b2, b3, b4])

        output_notop = self.conv2d_bn(ublock1, 128, 3, 3)
        model = Model(inputs=input_img, outputs=output_notop, name='MEDNet_notop')
        return model

    def _build_mednet(self):
        x = Input(self.input_shape)
        y = self.model_notop(x)
        output = Conv2D(filters=self.n_class, kernel_size=3, padding='same')(y)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        return Model(x, output)


class MEDNetV2:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

        self.model_notop = self._build_mednetv2_notop()
        self.model = self._build_mednetv2()

    def conv2d_bn(self, input_tensor, filters, kernel_size, padding='same', strides=(1, 1), dilation_rate=(1, 1),
                  use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001),
                  dtype=None,
                  batch_norm=True, activation=None, name=None):
        """Utility function to apply conv + BN.

        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            conv_name = name + '_conv2d'
            bn_name = name + '_bn'
            conv2d_name = name + '_conv2d'
            act_name = name + '_atv'
        else:
            bn_name = None
            conv_name = 'conv2d'
            act_name = None
            conv2d_name = None
        if backend.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = -1
        with tf.name_scope(name=conv_name):
            xi = Conv2D(
                filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                dilation_rate=dilation_rate,
                dtype=dtype,
                name=conv2d_name)(input_tensor)
            if batch_norm:
                xi = BatchNormalization(axis=bn_axis, scale=False, dtype=dtype, name=bn_name)(xi)
            if activation is not None:
                xi = Activation(activation, dtype=dtype, name=act_name)(xi)
        return xi

    def inception_v3_block_type1(self, input_tensor, filters=(64, 48, 64, 64, 96, 96, 32),
                                 name='inception_v3_block_type1'):
        with tf.name_scope(name=name):
            branch1x1 = self.conv2d_bn(input_tensor, filters[0], kernel_size=(
                1, 1), padding='same', activation='relu')

            branch3x3 = self.conv2d_bn(input_tensor, filters[1], kernel_size=(
                1, 1), padding='same', activation='relu')
            branch3x3 = self.conv2d_bn(branch3x3, filters[2], kernel_size=(
                3, 3), dilation_rate=(2,2) ,padding='same', activation='relu')

            branch3x3dbl = self.conv2d_bn(input_tensor, filters[3], kernel_size=(
                1, 1), padding='same', activation='relu')
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[4], kernel_size=(
                3, 3), padding='same', activation='relu')
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, filters[5], kernel_size=(
                3, 3), padding='same', activation='relu')

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(input_tensor)
            branch_pool = self.conv2d_bn(branch_pool, filters[6], kernel_size=(
                1, 1), padding='same', activation='relu')
            out = Concatenate()([branch1x1, branch3x3, branch3x3dbl, branch_pool])
        return out

    def resnet_identity_block(self, input_tensor, filters, kernel_size, stage, block_name=None,
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(0.001)):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
                   | ————————————
                   ↓                |
              conv2d BN relu        |
                   ↓                |
              conv2d BN relu        |
                   ↓                |
                conv2d BN           |
                  (+) ———————————
                   ↓
                 relu
        """
        filters1, filters2, filters3 = filters

        xi = self.conv2d_bn(input_tensor, filters1, kernel_size, padding='same', activation='relu',
                            kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                            name=block_name + str(stage) + '_conv2d_a')

        xi = self.conv2d_bn(xi, filters2, kernel_size, padding='same', activation='relu',
                            kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                            name=block_name + str(stage) + '_conv2d_b')

        # xi = conv2d_bn(xi, filters3, (1, 1), padding='same', activation=None,
        #                kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
        #                name=block_name + str(stage) + '_conv2d_c')

        with tf.name_scope(block_name + 'add' + str(stage)):
            xi = Add()([xi, input_tensor])
            xi = Activation('relu')(xi)
        return xi

    def resnet_conv_block(self, input_tensor, filters, kernel_size, stage, block_name, strides=(2, 2), batch_norm=True,
                          kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well


                   | —————————————————————
                   ↓                     |
            conv2d /2 BN relu            |
                   ↓                     ↓
              conv2d BN relu     conv2d /2 BN relu
                   ↓                     ↓
                conv2d BN                |
                  (+) ———————————————————
                   ↓
                 relu

        """
        filters1, filters2, filters3 = filters

        xi = self.conv2d_bn(input_tensor, filters1, kernel_size, strides=strides, activation='relu',
                            kernel_regularizer=kernel_regularizer, batch_norm=batch_norm,
                            kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_conv2d_a')

        xi = self.conv2d_bn(xi, filters2, kernel_size, padding='same', activation='relu',
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=kernel_initializer, batch_norm=batch_norm,
                            name=block_name + str(stage) + '_conv2d_b')

        shortcut = self.conv2d_bn(input_tensor, filters3, (1, 1), strides=strides, activation=None,
                                  batch_norm=batch_norm,
                                  kernel_regularizer=kernel_regularizer,
                                  kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_shortcut')

        with tf.name_scope(block_name + str(stage) + '_add'):
            xi = Add()([xi, shortcut])
            xi = Activation('relu')(xi)
        return xi

    def _build_mednetv2_notop(self):
        fts = [32, 48, 64, 128]
        inputs = Input(self.input_shape)
        with tf.name_scope('encoder_1'):
            x1 = self.inception_v3_block_type1(inputs, filters=(32, 24, 32, 32, 40, 40, 16),
                                               name='incept_b1')  # 304
        with tf.name_scope('encoder_2'):
            x = self.resnet_conv_block(x1, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=1,
                                       block_name='encoder_a')
            x = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=1,
                                           block_name='encoder_b')
            x = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=1,
                                           block_name='encoder_c')
            x2 = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=1,
                                            block_name='encoder_d')
        with tf.name_scope('encoder_3'):
            x = self.resnet_conv_block(x2, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=2,
                                       block_name='encoder_a')
            x = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=2,
                                           block_name='encoder_b')
            x = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=2,
                                           block_name='encoder_c')
            x3 = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=2,
                                            block_name='encoder_d')
        with tf.name_scope('mid'):
            x = self.resnet_conv_block(x3, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=3,
                                       block_name='mid_a')
            x = self.resnet_identity_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=3,
                                           block_name='mid_b')
            x = self.resnet_identity_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=3,
                                           block_name='mid_c')
            mid = Conv2DTranspose(fts[3], kernel_size=(3, 3), strides=2, padding='same')(x)
        with tf.name_scope('decoder_3'):
            x = Concatenate()([mid, x3])
            x = self.resnet_conv_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=4, strides=(1, 1),
                                       block_name='decoder_a')
            x = self.resnet_identity_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=4,
                                           block_name='decoder_b')
            x = self.resnet_identity_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=4,
                                           block_name='decoder_c')
            x = self.resnet_identity_block(x, filters=(fts[3], fts[3], fts[3]), kernel_size=(3, 3), stage=4,
                                           block_name='decoder_d')
            u3 = Conv2DTranspose(fts[2], kernel_size=(3, 3), strides=2, padding='same')(x)
        with tf.name_scope('decoder_2'):
            x = Concatenate()([u3, x2])
            x = self.resnet_conv_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=5, strides=(1, 1),
                                       block_name='decoder_a')
            x = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=5,
                                           block_name='decoder_b')
            x = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=5,
                                           block_name='decoder_c')
            x = self.resnet_identity_block(x, filters=(fts[2], fts[2], fts[2]), kernel_size=(3, 3), stage=5,
                                           block_name='decoder_d')
            u2 = Conv2DTranspose(fts[1], kernel_size=(3, 3), strides=2, padding='same')(x)
        with tf.name_scope('decoder_1'):
            x = Concatenate()([u2, x1])
            x = self.resnet_conv_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=6, strides=(1, 1),
                                       block_name='decoder_a')
            x = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=6,
                                           block_name='decoder_b')
            x = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=6,
                                           block_name='decoder_c')
            x = self.resnet_identity_block(x, filters=(fts[1], fts[1], fts[1]), kernel_size=(3, 3), stage=6,
                                           block_name='decoder_d')
        output_notop = self.conv2d_bn(x, 128, kernel_size=(3, 3), padding='same', activation='relu')
        model = Model(inputs=inputs, outputs=output_notop, name='MEDNetV2_notop')
        return model

    def _build_mednetv2(self):
        x = Input(self.input_shape)
        y = self.model_notop(x)
        output = Conv2D(filters=self.n_class, kernel_size=3, padding='same')(y)
        output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        return Model(x, output, name='MEDNetV2')


class UNet:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class

        self.model_notop = self._build_unet_notop()
        self.model = self._build_unet()

    def upsample_conv(self, filters, kernel_size, strides, padding):
        return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def upsample_simple(self, filters, kernel_size, strides, padding):
        return UpSampling2D(strides)

    def attention_gate(self, inp_1, inp_2, n_intermediate_filters):
        """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
           Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
        """
        inp_1_conv = Conv2D(
            n_intermediate_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
        )(inp_1)
        inp_2_conv = Conv2D(
            n_intermediate_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
        )(inp_2)

        f = Activation("relu")(Add()([inp_1_conv, inp_2_conv]))
        g = Conv2D(
            filters=1,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
        )(f)
        h = Activation("sigmoid")(g)
        return Multiply()([inp_1, h])

    def attention_concat(self, conv_below, skip_connection):
        """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
        """
        below_filters = conv_below.get_shape().as_list()[-1]
        attention_across = self.attention_gate(skip_connection, conv_below, below_filters)
        return Concatenate()([conv_below, attention_across])

    def conv2d_block(self,
                     inputs,
                     use_batch_norm=True,
                     dropout=0.3,
                     dropout_type="spatial",
                     filters=16,
                     kernel_size=(3, 3),
                     activation="relu",
                     kernel_initializer="he_normal",
                     padding="same"):
        if dropout_type == "spatial":
            DO = SpatialDropout2D
        elif dropout_type == "standard":
            DO = Dropout
        else:
            raise ValueError(
                f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
            )

        c = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=not use_batch_norm,
        )(inputs)
        if use_batch_norm:
            c = BatchNormalization()(c)
        if dropout > 0.0:
            c = DO(dropout)(c)
        c = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=not use_batch_norm,
        )(c)
        if use_batch_norm:
            c = BatchNormalization()(c)
        return c

    def _build_unet_notop(self,
                          activation="relu",
                          use_batch_norm=True,
                          upsample_mode="deconv",  # 'deconv' or 'simple'
                          dropout=0.3,
                          dropout_change_per_layer=0.0,
                          dropout_type="spatial",
                          use_dropout_on_upsampling=False,
                          use_attention=False,
                          filters=16,
                          num_layers=4
                          ):  # 'sigmoid' or 'softmax'

        """
        Customisable UNet architecture (Ronneberger et al. 2015 [1]).
        Arguments:
        input_shape: 3D Tensor of shape (x, y, num_channels)
        num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
        activation (str): A keras.activations.Activation to use. ReLu by default.
        use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
        upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
        dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
        dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
        dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
        use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
        use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
        filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
        num_layers (int): Number of total layers in the encoder not including the bottleneck layer
        output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
        Returns:
        model (keras.models.Model): The built U-Net
        Raises:
        ValueError: If dropout_type is not one of "spatial" or "standard"
        [1]: https://arxiv.org/abs/1505.04597
        [2]: https://arxiv.org/pdf/1411.4280.pdf
        [3]: https://arxiv.org/abs/1804.03999
        """

        if upsample_mode == "deconv":
            upsample = self.upsample_conv
        else:
            upsample = self.upsample_simple

        # Build U-Net model
        inputs = Input(self.input_shape)
        x = inputs

        down_layers = []
        for l in range(num_layers):
            x = self.conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters = filters * 2  # double the number of filters with each layer

        x = self.conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0

        for conv in reversed(down_layers):
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            if use_attention:
                x = self.attention_concat(conv_below=x, skip_connection=conv)
            else:
                x = Concatenate()([x, conv])
            x = self.conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
        model = Model(inputs=[inputs], outputs=[x], name='unet_notop')
        return model

    def _build_unet(self):
        x = Input(self.input_shape)
        y = self.model_notop(x)
        outputs = Conv2D(self.n_class, (1, 1), activation='softmax')(y)
        return Model(x, outputs, name='unet')


# Bai et al.
class FCN8_VGG16:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = self.build_fcn8()

    # crop o1 wrt o2
    def crop(self, o1, o2, i):

        o_shape2 = Model(i, o2).output_shape
        output_height2 = o_shape2[1]
        output_width2 = o_shape2[2]

        o_shape1 = Model(i, o1).output_shape
        output_height1 = o_shape1[1]
        output_width1 = o_shape1[2]

        cx = abs(output_width1 - output_width2)
        cy = abs(output_height2 - output_height1)

        if output_width1 > output_width2:
            o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

        if output_height1 > output_height2:
            o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

        return o1, o2

    def get_vgg_encoder(self, input_shape, pretrained='imagenet'):
        assert input_shape[0] % 32 == 0
        assert input_shape[1] % 32 == 0
        pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                         "releases/download/v0.1/" \
                         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

        img_input = Input(shape=input_shape)

        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        f1 = x
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same',
                   name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        f3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        f4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        f5 = x

        if pretrained == 'imagenet':
            VGG_Weights_path = tf.keras.utils.get_file(
                pretrained_url.split("/")[-1], pretrained_url)
            Model(img_input, x).load_weights(VGG_Weights_path)

        return img_input, [f1, f2, f3, f4, f5]

    def build_fcn8(self):
        img_input, levels = self.get_vgg_encoder(input_shape=self.input_shape)
        [f1, f2, f3, f4, f5] = levels

        o = f5

        o = (Conv2D(4096, (7, 7), activation='relu',
                    padding='same'))(o)
        # o = Dropout(0.5)(o)
        o = (Conv2D(4096, (1, 1), activation='relu',
                    padding='same'))(o)
        # o = Dropout(0.5)(o)

        # o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal'))(o)
        o = (Conv2D(21, (1, 1), kernel_initializer='he_normal'))(o)
        # o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        o = Conv2DTranspose(21, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)

        o2 = f4
        o2 = (Conv2D(21, (1, 1), kernel_initializer='he_normal'))(o2)
        o, o2 = self.crop(o, o2, img_input)

        o = Add()([o, o2])

        o = Conv2DTranspose(21, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        o2 = f3
        o2 = (Conv2D(21, (1, 1), kernel_initializer='he_normal'))(o2)
        o2, o = self.crop(o2, o, img_input)
        o = Add()([o2, o])

        o = Conv2DTranspose(21, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
        o = (Conv2D(self.n_class, (1, 1), kernel_initializer='he_normal'))(o)
        o = Cropping2D(cropping=((0, 8), (0, 8)))(o)
        o = (Activation('softmax'))(o)
        model = Model(img_input, o, name='fcn8_vgg16')
        return model


# Schlegl et al
class Encoder_Decoder:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class
        self.model = self._build_encoder_decoder()

    def _build_encoder_decoder(self):
        inputs = Input(self.input_shape)

        x = Conv2D(16, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(self.n_class, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        return Model(inputs, x)


# Girish et al
class FCN_Unet:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)

        x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x)

        x = MaxPooling2D()(x1)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x2 = BatchNormalization()(x)

        x = MaxPooling2D()(x2)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x3 = BatchNormalization()(x)

        x = MaxPooling2D()(x3)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x4 = BatchNormalization()(x)

        x = MaxPooling2D()(x4)
        x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        xm = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)

        x = Concatenate()([x4, xm])
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        xd3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)

        x = Concatenate()([x3, xd3])
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        xd2 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)

        x = Concatenate()([x2, xd2])
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        xd1 = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same')(x)

        x = Concatenate()([x1, xd1])
        x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        out = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(x)

        return Model(inputs, out)


# Li et al
class Unet3D:
    def __init__(self, input_shape, n_class):
        self.input_shape = input_shape
        self.n_class = n_class
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x1 = ReLU()(x)

        x = MaxPooling3D(pool_size=(2, 2, 1))(x1)
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x2 = ReLU()(x)

        x = MaxPooling3D(pool_size=(2, 2, 1))(x2)
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x3 = ReLU()(x)

        x = MaxPooling3D(pool_size=(2, 2, 1))(x3)
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x4 = ReLU()(x)

        x = MaxPooling3D(pool_size=(2, 2, 1))(x4)
        x = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        xd = ReLU()(x)

        x = Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(xd)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, x4])
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        xu1 = ReLU()(x)

        x = Conv3DTranspose(16, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(xu1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, x3])
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        xu2 = ReLU()(x)

        x = Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(xu2)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, x2])
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        xu3 = ReLU()(x)

        x = Conv3DTranspose(4, kernel_size=(3, 3, 3), strides=(2, 2, 1), padding='same')(xu3)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, x1])
        x = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(3, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        output = Activation('softmax')(x)

        return Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # MEDNetV2((400, 400, 1), n_class=3).model.summary()
    # UNet((400,400,1),n_class=3).model_notop.summary()
    # Unet3D((512, 512, 512, 1), n_class=3).model.summary()
    # MEDNet((64, 64, 1), n_class=3).model.summary()
    model = DenseNet121()
    model.save('dense.h5')
