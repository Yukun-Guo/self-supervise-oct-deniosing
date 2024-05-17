"""
utility blocks for network
Version 1.1
Yukun Guo
CEI, OHSU
"""

import tensorflow as tf
from keras import layers, regularizers, backend, models
from keras import backend as K


class ProjectionEnface(layers.Layer):
    def __init__(self, output_slab_number, slab_position_initializer='uniform', **kwargs):
        self.slab_position_initializer = slab_position_initializer
        self.ouput_slab_number = output_slab_number
        self.slab_positions = None
        super(ProjectionEnface, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        shape_seg, _ = input_shape
        # Create a trainable weight variable for this layer.
        self.slab_positions = self.add_weight(name='slab_position',
                                              shape=(
                                                  shape_seg[-1], self.ouput_slab_number),
                                              initializer='uniform',
                                              trainable=True)
        # Be sure to call this at the end
        super(ProjectionEnface, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        seg, octa = inputs
        slab_ps = tf.unstack(self.slab_positions, axis=1)
        hod = []
        for i, tr in enumerate(slab_ps):
            # weight the segmentation mask
            seg_w = seg * (tr / tf.reduce_sum(tr))
            seg_sum = tf.reduce_sum(seg_w, axis=-1)
            octa_w = tf.squeeze(octa, axis=-1) * seg_sum
            slabs = tf.reduce_mean(octa_w, 1)
            slabs = (slabs - tf.reduce_min(slabs)) / \
                (tf.reduce_max(slabs) - tf.reduce_min(slabs))
            slabs = tf.expand_dims(slabs, 3)
            hod.append(slabs)
        return [hod[0], hod[1], hod[2]]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_seg, _ = input_shape
        return [(shape_seg[0], shape_seg[2], shape_seg[3], 1), (shape_seg[0], shape_seg[2], shape_seg[3], 1),
                (shape_seg[0], shape_seg[2], shape_seg[3], 1)]


class CustomRegularization(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def call(self, x, mask=None):
        bce = tf.losses.binary_crossentropy(x[0], x[1])
        loss2 = K.sum(bce)
        self.add_loss(loss2, x)
        # you can output whatever you need, just update output_shape adequately
        # But this is probably useful
        return bce

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


def conv2d_bn(input_tensor, filters, kernel_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), use_bias=False,
              kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), dtype=None,
              use_batchnorm=True, activation=None, name=None):
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
        xi = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            dtype=dtype,
            name=conv2d_name)(input_tensor)
        if use_batchnorm:
            xi = layers.BatchNormalization(
                axis=bn_axis, scale=False, dtype=dtype, name=bn_name)(xi)
        if activation is not None:
            xi = layers.Activation(activation, dtype=dtype, name=act_name)(xi)
    return xi



def conv3d_bn(input_tensor, filters, kernel_size, padding='same', strides=(1, 1, 1), dilation_rate=(1, 1, 1),
              kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), activation=None, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv3d'
        bn_name = name + '_bn'
        conv3d_name = name + '_conv3d'
        act_name = name + '_atv'
    else:
        bn_name = None
        conv_name = 'conv3d'
        act_name = None
        conv3d_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    with tf.name_scope(name=conv_name):
        xi = layers.Conv3D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name=conv3d_name)(input_tensor)
        xi = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(xi)
        if activation is not None:
            xi = layers.Activation(activation, name=act_name)(xi)
    return xi



def separableconv2d_bn(input_tensor, filters, kernel_size, depth_multiplier=1, padding='same', strides=(1, 1),
                       dilation_rate=(1, 1), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001),
                       activation=None, name=None):
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
        conv_name = name + '_seperableconv2d'
        bn_name = name + '_bn'
        conv2d_name = name + '_seperableconv2d'
        act_name = name + '_atv'
    else:
        bn_name = None
        conv_name = 'seperableconv2d'
        act_name = None
        conv2d_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    with tf.name_scope(name=conv_name):
        xi = layers.SeparableConv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name=conv2d_name)(input_tensor)
        xi = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(xi)
        if activation is not None:
            xi = layers.Activation(activation, name=act_name)(xi)
    return xi


def mcconv_2d(input_tensor, filters, kernel_sizes, depth_multipliers, padding='same', strides=(1, 1),
              dilation_rate=(1, 1), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001),
              activation=None, name=None):
    if name is not None:
        conv_name = name + '_mcconv_2d'
        bn_name = name + '_bn'
        conv2d_name = name + '_mcconv_2d'
        act_name = name + '_atv'
    else:
        bn_name = None
        conv_name = 'mcconv_2d'
        act_name = None
        conv2d_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    with tf.name_scope(name=conv_name):
        G = len(filters)
        y = []
        for xi, fi, kernels, mti in zip(tf.split(input_tensor, G, axis=bn_axis), filters, kernel_sizes,
                                        depth_multipliers):
            xot = layers.SeparableConv2D(
                filters, kernels,
                strides=strides,
                padding=padding,
                depth_multiplier=mti,
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                dilation_rate=dilation_rate,
                name=conv2d_name)(input_tensor)
            y.append(xot)
        xot = layers.Concatenate()(y)
        xot = layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(xot)
        if activation is not None:
            xot = layers.Activation(activation, name=act_name)(xot)
    return xot


def resnet_identity_block(input_tensor, filters, kernel_size, stage, block_name, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.001)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 2 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
               | ————————————
               ↓                |
          conv2d BN relu        |
               ↓                |
            conv2d BN           |
              (+) ———————————
               ↓
             relu
    """
    filters1, filters2 = filters

    xi = conv2d_bn(input_tensor, filters1, kernel_size, padding='same', activation='relu',
                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                   name=block_name + str(stage) + '_conv2d_a')

    xi = conv2d_bn(xi, filters2, kernel_size, padding='same', activation=None,
                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                   name=block_name + str(stage) + '_conv2d_b')

    with tf.name_scope(block_name + 'add' + str(stage)):
        xi = layers.add([xi, input_tensor])
        xi = layers.Activation('relu')(xi)
    return xi



def resnet_identity_block_3d(input_tensor, filters, kernel_size, stage, block_name, kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(0.001)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 2 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
               | ————————————
               ↓                |
          conv3d BN relu        |
               ↓                |
            conv3d BN           |
              (+) ———————————
               ↓
             relu
    """

    xi = conv3d_bn(input_tensor, filters, kernel_size, padding='same', activation='relu',
                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                   name=block_name + str(stage) + '_conv3d_a')

    xi = conv3d_bn(xi, filters, kernel_size, padding='same', activation=None,
                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer,
                   name=block_name + str(stage) + '_conv3d_b')

    with tf.name_scope(block_name + 'add' + str(stage)):
        xi = layers.add([xi, input_tensor])
        xi = layers.Activation('relu')(xi)
    return xi


def resnet_shortcut_block(input_tensor, filters, kernel_size, stage, block_name, strides=(2, 2), use_batchnorm=True,
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
               |                conv2d /2 BN
               ↓                     ↓
            conv2d BN                |
              (+) ———————————————————
               ↓
             relu

    """
    filters1, filters2 = filters

    xi = conv2d_bn(input_tensor, filters1, kernel_size, strides=strides, activation='relu',
                   kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm,
                   kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_conv2d_a')

    xi = conv2d_bn(xi, filters2, kernel_size, padding='same', activation=None, kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, use_batchnorm=use_batchnorm,
                   name=block_name + str(stage) + '_conv2d_b')

    shortcut = conv2d_bn(input_tensor, filters2, (1, 1), strides=strides, activation=None, use_batchnorm=use_batchnorm,
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_shortcut')

    with tf.name_scope(block_name + str(stage) + '_add'):
        xi = layers.add([xi, shortcut])
        xi = layers.Activation('relu')(xi)
    return xi



def resnet_shortcut_block_3d(input_tensor, filters, kernel_size, stage, block_name, strides=(2, 2, 2),
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
    the first conv layer at main path is with strides=(2, 2, 2)
    And the shortcut should have strides=(2, 2, 2) as well


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

    xi = conv3d_bn(input_tensor, filters, kernel_size, strides=strides, activation='relu',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_conv3d_a')

    xi = conv3d_bn(xi, filters, kernel_size, padding='same', activation=None, kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_conv3d_b')

    shortcut = conv3d_bn(input_tensor, filters, (1, 1, 1), strides=strides, activation=None,
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer, name=block_name + str(stage) + '_shortcut')

    with tf.name_scope(block_name + str(stage) + '_add'):
        xi = layers.add([xi, shortcut])
        xi = layers.Activation('relu')(xi)
    return xi


def resnet_pre_activation_block(input_tensor, filters, kernel_size, stage, block_name,
                                kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)):
    '''
        Output
            ↑
           Add <----------+
            ↑             |
        3x3 Conv          |
            ↑             |
           ReLu           |
            ↑             +
            BN        1x1 Conv
            ↑             ^
        3x3 Conv          |
            ↑             |
           ReLu           |
            ↑             |
            BN            |
            +-------------+
            ↑
          input
    '''
    with tf.name_scope(block_name + str(stage) + '_pre_active'):
        filter1, filter2 = filters
        x_i = layers.Conv2D(filters=filter2, kernel_size=(1, 1), padding='same',
                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_tensor)
        x_p = layers.BatchNormalization()(input_tensor)
        x_p = layers.Activation('relu')(x_p)
        x_p = layers.Conv2D(filters=filter1, kernel_size=kernel_size, padding='same',
                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_p)
        x_p = layers.BatchNormalization()(x_p)
        x_p = layers.Activation('relu')(x_p)
        x_p = layers.Conv2D(filters=filter2, kernel_size=kernel_size, padding='same',
                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_p)
        x_p = layers.Add()([x_p, x_i])
    return x_p


def inception_v3_input(input_tensor, name='input_inception_v3'):
    with tf.name_scope(name):
        xi = conv2d_bn(input_tensor, 32, (3, 3), strides=(
            2, 2), activation='relu', padding='valid')
        xi = conv2d_bn(xi, 32, (3, 3), activation='relu', padding='valid')
        xi = conv2d_bn(xi, 64, (3, 3), activation='relu', padding='same')
        xi = layers.MaxPooling2D((3, 3), strides=(2, 2))(xi)

        xi = conv2d_bn(xi, 80, (1, 1), activation='relu', padding='valid')
        xi = conv2d_bn(xi, 192, (1, 1), activation='relu', padding='valid')
        out = layers.MaxPooling2D((3, 3), strides=(2, 2))(xi)
    return out


def inception_v3_input(input_tensor, name='input_inception_v3_2D'):
    with tf.name_scope(name):
        x1 = conv2d_bn(input_tensor, 32, (3, 3),
                       activation='relu', padding='same')
        x1 = conv2d_bn(x1, 32, (3, 3), activation='relu', padding='same')

        x2 = conv2d_bn(input_tensor, 16, (3, 3),
                       activation='relu', padding='same')
        x3 = conv2d_bn(input_tensor, 16, (1, 1),
                       activation='relu', padding='same')
        out = layers.Concatenate()([x1, x2, x3])
    return out


def inception_v3_input_3d(input_tensor, name='input_inception_v3_3D'):
    with tf.name_scope(name):
        x1 = conv3d_bn(input_tensor, 32, (3, 3, 3),
                       activation='relu', padding='same')
        x1 = conv3d_bn(x1, 32, (3, 3, 3), activation='relu', padding='same')

        x2 = conv3d_bn(input_tensor, 16, (3, 3, 3),
                       activation='relu', padding='same')
        x3 = conv3d_bn(input_tensor, 16, (1, 1, 1),
                       activation='relu', padding='same')
        out = layers.Concatenate()([x1, x2, x3])
    return out


def inception_v3_block_type1(input_tensor, name, filters=(64, 48, 64, 64, 96, 96, 32)):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    with tf.name_scope(name=name):
        branch1x1 = conv2d_bn(input_tensor, filters[0], kernel_size=(
            1, 1), padding='same', activation='relu')

        branch5x5 = conv2d_bn(input_tensor, filters[1], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch5x5 = conv2d_bn(branch5x5, filters[2], kernel_size=(
            5, 5), padding='same', activation='relu')

        branch3x3dbl = conv2d_bn(input_tensor, filters[3], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch3x3dbl = conv2d_bn(branch3x3dbl, filters[4], kernel_size=(
            3, 3), padding='same', activation='relu')
        branch3x3dbl = conv2d_bn(branch3x3dbl, filters[5], kernel_size=(
            3, 3), padding='same', activation='relu')

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch_pool = conv2d_bn(branch_pool, filters[6], kernel_size=(
            1, 1), padding='same', activation='relu')
        out = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)
    return out


def inception_v3_block_type1_3d(input_tensor, name, filters=(64, 48, 64, 64, 96, 96, 32)):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    with tf.name_scope(name=name):
        branch1x1 = conv3d_bn(input_tensor, filters[0], kernel_size=(
            1, 1, 1), padding='same', activation='relu')

        branch5x5 = conv3d_bn(input_tensor, filters[1], kernel_size=(
            1, 1, 1), padding='same', activation='relu')
        branch5x5 = conv3d_bn(branch5x5, filters[2], kernel_size=(
            5, 5, 5), padding='same', activation='relu')

        branch3x3dbl = conv3d_bn(input_tensor, filters[3], kernel_size=(
            1, 1, 1), padding='same', activation='relu')
        branch3x3dbl = conv3d_bn(branch3x3dbl, filters[4], kernel_size=(
            3, 3, 3), padding='same', activation='relu')
        branch3x3dbl = conv3d_bn(branch3x3dbl, filters[5], kernel_size=(
            3, 3, 3), padding='same', activation='relu')

        branch_pool = layers.AveragePooling3D(
            (3, 3, 3), strides=(1, 1, 1), padding='same')(input_tensor)
        branch_pool = conv3d_bn(branch_pool, filters[6], kernel_size=(
            1, 1, 1), padding='same', activation='relu')
        out = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)
    return out


def inception_v3_block_type2(input_tensor, name, filters=(384, 64, 96, 96)):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    with tf.name_scope(name=name):
        branch3x3 = conv2d_bn(input_tensor, filters[0], kernel_size=(3, 3), strides=(2, 2), padding='valid',
                              activation='relu')

        branch3x3dbl = conv2d_bn(input_tensor, filters[1], kernel_size=(
            1, 1), padding='valid', activation='relu')
        branch3x3dbl = conv2d_bn(branch3x3dbl, filters[2], kernel_size=(
            3, 3), padding='same', activation='relu')
        branch3x3dbl = conv2d_bn(branch3x3dbl, filters[3], kernel_size=(3, 3), strides=(2, 2), padding='valid',
                                 activation='relu')

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(input_tensor)
        out = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis)
    return out


def inception_v3_block_type3(input_tensor, name, filters=(192, 128, 128, 192, 128, 128, 128, 128, 192, 192)):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    with tf.name_scope(name=name):
        branch1x1 = conv2d_bn(input_tensor, filters[0], kernel_size=(
            1, 1), padding='same', activation='relu')

        branch7x7 = conv2d_bn(input_tensor, filters[1], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch7x7 = conv2d_bn(branch7x7, filters[2], kernel_size=(
            1, 7), padding='same', activation='relu')
        branch7x7 = conv2d_bn(branch7x7, filters[3], kernel_size=(
            7, 1), padding='same', activation='relu')

        branch7x7dbl = conv2d_bn(input_tensor, filters[4], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch7x7dbl = conv2d_bn(branch7x7dbl, filters[5], kernel_size=(
            7, 1), padding='same', activation='relu')
        branch7x7dbl = conv2d_bn(branch7x7dbl, filters[6], kernel_size=(
            1, 7), padding='same', activation='relu')
        branch7x7dbl = conv2d_bn(branch7x7dbl, filters[7], kernel_size=(
            7, 1), padding='same', activation='relu')
        branch7x7dbl = conv2d_bn(branch7x7dbl, filters[8], kernel_size=(
            1, 7), padding='same', activation='relu')

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch_pool = conv2d_bn(branch_pool, filters[9], kernel_size=(
            1, 1), padding='same', activation='relu')
        out = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)
    return out


def inception_v3_block_type4(input_tensor, name, filters=(320, 384, 384, 384, 448, 384, 384, 384, 192)):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    with tf.name_scope(name=name):
        branch1x1 = conv2d_bn(input_tensor, filters[0], kernel_size=(
            1, 1), padding='same', activation='relu')

        branch3x3 = conv2d_bn(input_tensor, filters[1], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch3x3_1 = conv2d_bn(branch3x3, filters[2], kernel_size=(
            1, 3), padding='same', activation='relu')
        branch3x3_2 = conv2d_bn(branch3x3, filters[3], kernel_size=(
            3, 1), padding='same', activation='relu')
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis)

        branch3x3dbl = conv2d_bn(input_tensor, filters[4], kernel_size=(
            1, 1), padding='same', activation='relu')
        branch3x3dbl = conv2d_bn(branch3x3dbl, filters[5], kernel_size=(
            3, 3), padding='same', activation='relu')
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, filters[6], kernel_size=(
            1, 3), padding='same', activation='relu')
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, filters[7], kernel_size=(
            3, 1), padding='same', activation='relu')

        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch_pool = conv2d_bn(branch_pool, filters[8], kernel_size=(
            1, 1), padding='same', activation='relu')
        out = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis)
    return out


def inception_v1_a(x, filters=(16, 32, 64, 96, 128,), kernel_size=(1, 3, 5), padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_v1_a'):
    #                     Concat Output
    #                           ↑
    #     +----------------------------------------+
    #     |             ↑               ↑          ↑
    #     |         1x1 Conv        3x3 Conv   5x5 Conv
    #     |             ↑               ↑          ↑
    # 1x1 Conv   3x3/2 MaxPooling   1x1 Conv   1x1 Conv
    #     |             |               |          |
    #     +-------------+---------------+----------+
    #                            ↑
    #                          input

    x1 = conv2d_bn(x, filters[2], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1')
    x3_reduce = conv2d_bn(x, filters[3], kernel_size=kernel_size[1], padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_reduce')
    x3_pad = layers.ZeroPadding2D(padding=(1, 1))(x3_reduce)
    x3 = conv2d_bn(x3_pad, filters[4], kernel_size=kernel_size[1], padding='valid',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3')
    x5_reduce = conv2d_bn(x, filters[0], kernel_size=kernel_size[2], padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x5_reduce')
    x5_pad = layers.ZeroPadding2D(padding=(2, 2))(x5_reduce)
    x5 = conv2d_bn(x5_pad, filters[1], kernel_size=kernel_size[2], padding='valid',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x5')
    p = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same', name='pool')(x)
    pool_proj = conv2d_bn(p, filters[1], kernel_size=kernel_size[0], padding='same',
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'pool_proj')

    out = layers.Concatenate()([x1, x3, x5, pool_proj])
    return out


def inception_v2_a(x, filters=(16, 32, 64, 96, 128), kernel_size=(1, 3), padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_v2_a'):
    #                     Concat Output
    #                           ↑
    #     +----------------------------------------+
    #     |             ↑               ↑          ↑
    #     |             |               |      3x3 Conv
    #     |             |               |          ↑
    #     |          1x1 Conv        3x3 Conv  3x3 Conv
    #     |             ↑               ↑          ↑
    # 1x1 Conv   3x3/2 MaxPooling   1x1 Conv   1x1 Conv
    #     |             |               |          |
    #     +-------------+---------------+----------+
    #                            ↑
    #                          input

    x1 = conv2d_bn(x, filters[2], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1')
    x3_reduce = conv2d_bn(x, filters[3], kernel_size=kernel_size[1], padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_reduce')
    x3_pad = layers.ZeroPadding2D(padding=(1, 1))(x3_reduce)
    x3 = conv2d_bn(x3_pad, filters[4], kernel_size=kernel_size[1], padding='valid',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3')
    x1_1_reduce = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], padding=padding,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer, activation=activation,
                            name=name + 'double_x3_reduce')
    double_x1_pad = layers.ZeroPadding2D(padding=(2, 2))(x1_1_reduce)
    double_x3 = conv2d_bn(double_x1_pad, filters[1], kernel_size=kernel_size[1], padding='valid',
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'double_x3')
    double_x3_3 = conv2d_bn(double_x3, filters[1], kernel_size=kernel_size[1], padding='valid',
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'double_x3_3')
    p = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same', name='pool')(x)
    pool_proj = conv2d_bn(p, filters[1], kernel_size=kernel_size[0], padding='same',
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'pool_proj')

    out = layers.Concatenate()([x1, x3, double_x3_3, pool_proj])
    return out


def inception_v2_b(x, filters=(128, 96, 64), kernel_size=3, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_v2_b'):
    #                         Concat Output
    #                               ↑
    #    +------------------------------------------------------+
    #    ↑           ↑           ↑          ↑         ↑         ↑
    #    |           |           |          |       3x1 Conv  1x3 Conv
    #    |           |           |          |           ↑         ↑
    #    |        1x1 Conv    1x3 Conv    3x1 Conv        3x3 Conv
    #    |                         ↑        ↑                ↑
    # 1x1 Conv   Avg Pooling        1x1 Conv              1x1 Conv
    #    ↑            ↑                 ↑                    ↑
    #    +---------------------------------------------------+
    #                                ↑
    #                              input

    x1_1 = conv2d_bn(x, filters[0], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=(kernel_size, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x1_4 = conv2d_bn(x1_2, filters[0], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_4')

    x2_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[1], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x2_3 = conv2d_bn(x2_1, filters[1], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_3')
    x3_1 = layers.AveragePooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same', name='pool')(x)
    x3_2 = conv2d_bn(x3_1, filters[2], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_2')
    x4_1 = conv2d_bn(x, filters[2], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x4_1')
    out = layers.Concatenate()([x1_3, x1_4, x2_2, x2_3, x3_2, x4_1])
    return out


def inception_v4_a(x, filters=(64, 96), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), use_batchnorm=True, activation='relu', name='inception_v4_a'):
    #                     Concat Output
    #                        ↑
    #    +------------------------------------+
    #    ↑           ↑            ↑           ↑
    #    |           |            |       3x3 Conv
    #    |           |            |           ↑
    #    |        1x1 Conv     3x3 Conv   3x3 Conv
    #    |                         ↑          ↑
    # 1x1 Conv   Avg Pooling   1x1 Conv   1x1 Conv
    #    ↑            ↑            ↑          ↑
    #    +------------------------------------+
    #                        ↑
    #                      input
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm, activation=activation,
                     name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=kernel_size[1], padding=padding, use_batchnorm=use_batchnorm,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[1], kernel_size=kernel_size[1], padding=padding, use_batchnorm=use_batchnorm,
                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                     activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm, activation=activation,
                     name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[1], kernel_size=kernel_size[1], padding=padding, use_batchnorm=use_batchnorm,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm, activation=activation,
                     name=name + 'x3_1')
    x4_1 = layers.AveragePooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same', name='pool')(x)
    x4_2 = conv2d_bn(x4_1, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm, activation=activation,
                     name=name + 'x4_2')
    out = layers.Concatenate()([x1_3, x2_2, x3_1, x4_2])
    return out


def reduction_v4_a(x, filters=(64, 96), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_v4_a'):
    #                    Concat Output
    #                       ↑
    #         +-------------+----------+
    #         |             |          |
    #         |             |     3x3/2 Conv
    #         |             |          ↑
    #         |             |       3x3 Conv
    #         |             |          ↑
    # 3x3/2 MaxPooling  3x3/2 Conv  1x1 Conv
    #         ↑             ↑          ↑
    #         +------------------------+
    #                       ↑
    #                     input

    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[1], kernel_size=kernel_size[1], padding=padding, strides=2,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], padding=padding, strides=2,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x3_1 = layers.MaxPool2D(pool_size=(3, 3), strides=(
        2, 2), padding='same', name='pool')(x)

    out = layers.Concatenate()([x1_3, x2_1, x3_1])
    return out


def inception_v4_b(x, filters=(128, 192, 224, 256, 384), kernel_size=7, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_v4_a'):
    #                Concat Output
    #                      ↑
    #     +----------+---------+---------+
    #     |          |         |      7x1 Conv
    #     |          |         |         ↑
    #     |          |         |      1x7 Conv
    #     |          |         |         ↑
    #     |          |      7x1 Conv  7x1 Conv
    #     |          |         ↑         ↑
    #     |      1x1 Conv   1x7 Conv  1x7 Conv
    #     |          ↑         ↑         ↑
    # 1x1 Conv  Avg Pooling 1x1 Conv  1x1 Conv
    #     ↑          ↑         ↑         ↑
    #     +------------------------------+
    #                      ↑
    #                    input

    x1_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x1_4 = conv2d_bn(x1_3, filters[2], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_4')
    x1_5 = conv2d_bn(x1_4, filters[3], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_5')
    x2_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[3], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x2_3 = conv2d_bn(x2_2, filters[3], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_3')
    x3_1 = layers.AveragePooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same', name='pool')(x)
    x3_2 = conv2d_bn(x3_1, filters[0], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_2')
    x4_1 = conv2d_bn(x, filters[3], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x4_1')
    out = layers.Concatenate()([x1_5, x2_3, x3_2, x4_1])
    return out


def reduction_v4_b(x, filters=(192, 256, 320), n=7, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_v4_b'):
    #                Concat Output
    #                       ↑
    #         +-------------+----------+
    #         |             |    3x3/2 Conv
    #         |             |          ↑
    #         |             |      7x1 Conv
    #         |             |          ↑
    #         |       3x3/2 Conv   1x7 Conv
    #         |             ↑          ↑
    # 3x3/2 MaxPooling  1x1 Conv   1x1 Conv
    #         ↑             ↑          ↑
    #         +------------------------+
    #                       ↑
    #                     input

    x1_1 = conv2d_bn(x, filters[1], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=(kernel_size[0], n), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=(n, kernel_size[0]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x1_4 = conv2d_bn(x1_3, filters[2], kernel_size=kernel_size[1], padding=padding, strides=2,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_4')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[0], kernel_size=kernel_size[1], padding=padding, strides=2,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = layers.MaxPool2D(pool_size=(3, 3), strides=(
        2, 2), padding='same', name='pool')(x)
    out = layers.Concatenate()([x1_4, x2_2, x3_1])
    return out


def inception_v4_c(x, filters=(256, 384, 448, 512), kernel_size=3, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_v4_c'):
    #                            Concat Output
    #                                  ↑
    #    +-----------+-----------+----------+---------+---------+
    #    |           |           |          |         |         |
    #    |           |           |          |       3x1 Conv  1x3 Conv
    #    |           |           |          |           ↑         ↑
    #    |           |           |          |             3x1 Conv
    #    |           |           |          |                ↑
    #    |        1x1 Conv    1x3 Conv    3x1 Conv        1x3 Conv
    #    |           ↑             ↑        ↑                ↑
    # 1x1 Conv   Avg Pooling        1x1 Conv              1x1 Conv
    #    ↑           ↑                 ↑                     ↑
    #    +---------------------------------------------------+
    #                                ↑
    #                              input

    x1_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[2], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[3], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x1_4 = conv2d_bn(x1_3, filters[0], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_4')
    x1_5 = conv2d_bn(x1_3, filters[0], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_5')

    x2_1 = conv2d_bn(x, filters[1], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[0], kernel_size=(1, kernel_size), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x2_3 = conv2d_bn(x2_1, filters[0], kernel_size=(kernel_size, 1), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_3')
    x3_1 = layers.AveragePooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same', name='pool')(x)
    x3_2 = conv2d_bn(x3_1, filters[0], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_2')
    x4_1 = conv2d_bn(x, filters[0], kernel_size=(1, 1), padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x4_1')
    out = layers.Concatenate()([x1_4, x1_5, x2_2, x2_3, x3_2, x4_1])
    return out


def inception_resnet_v1_a(x, filters=(32, 256), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v1_a'):
    #               Output
    #                 ↑
    #                Add ←---------------+
    #                 ↑                  |
    #              1x1 Conv              |
    #                 ↑                  |
    #           Concat Output            |
    #                 ↑                  |
    #    +--------------------------     |
    #    ↑            ↑            ↑     |
    #    |            |        3x3 Conv  |
    #    |            |            ↑     |
    #    |         3x3 Conv    3x3 Conv  |
    #    |            ↑            ↑     |
    # 1x1 Conv     1x1 Conv    1x1 Conv  |
    #    ↑            ↑            ↑     |
    #    +-------------------------------+
    #                     ↑
    #                   input

    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_1')
    x0_0 = layers.Concatenate()([x1_3, x2_2, x3_1])
    x0_1 = conv2d_bn(x0_0, filters[1], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, name=name + 'x0_1')
    out = layers.Add()([x, x0_1])
    return out


def inception_resnet_v1_b(x, filters=(128, 896), kernel_size=(1, 7), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v1_b'):
    #          Output
    #            ↑
    #           Add ←--------+
    #            ↑           |
    #         1x1 Conv       |
    #            ↑           |
    #          Concat        |
    #            ↑           |
    #       +-----------+    |
    #       ↑           ↑    |
    #       |       1x7 Conv |
    #       |           ↑    |
    #       |       7x1 Conv |
    #       |           ↑    |
    #    1x1 Conv   1x1 Conv |
    #       ↑           ↑    |
    #       +----------------+
    #              ↑
    #            input

    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=(kernel_size[0], kernel_size[1]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=(kernel_size[1], kernel_size[0]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x0_0 = layers.Concatenate()([x1_3, x2_1])
    x0_1 = conv2d_bn(x0_0, filters[1], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=None, name=name + 'x3_2')
    out = layers.Add()([x, x0_1])
    return out


def inception_resnet_v1_c(x, filters=(192, 1792), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v1_c'):
    #          Output
    #            ↑
    #           Add ←--------+
    #            ↑           |
    #         1x1 Conv       |
    #            ↑           |
    #          Concat        |
    #            ↑           |
    #       +-----------+    |
    #       ↑           ↑    |
    #       |       1x3 Conv |
    #       |           ↑    |
    #       |       3x1 Conv |
    #       |           ↑    |
    #    1x1 Conv   1x1 Conv |
    #       ↑           ↑    |
    #       +----------------+
    #              ↑
    #            input

    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=(kernel_size[0], kernel_size[1]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=(kernel_size[1], kernel_size[0]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x0_0 = layers.Concatenate()([x1_3, x2_1])
    x0_1 = conv2d_bn(x0_0, filters[1], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=None, name=name + 'x3_2')
    out = layers.Add()([x, x0_1])
    return out


def reduction_resnet_v1_a(x, filters=(256, 384), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_resnet_v1_a'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[1], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x3_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(
        2, 2), padding=padding, name='pool')(x)
    out = layers.Concatenate()([x1_3, x2_1, x3_1])
    return out


def reduction_resnet_v1_b(x, filters=(256, 384), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_resnet_v1_b'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_1')
    x3_2 = conv2d_bn(x3_1, filters[1], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_2')
    x4_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(
        2, 2), padding=padding, name='pool')(x)
    out = layers.Concatenate()([x1_3, x2_2, x3_2, x4_1])
    return out


def inception_resnet_v2_a(x, filters=(32, 48, 64, 384), kernel_size=(1, 3), padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v2_a'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_1')
    x0_0 = layers.Concatenate()([x1_3, x2_2, x3_1])
    x0_1 = conv2d_bn(x0_0, filters[3], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, name=name + 'x0_1')
    out = layers.Add()([x, x0_1])
    return out


def inception_resnet_v2_b(x, filters=(128, 160, 192, 1154), kernel_size=(1, 7), padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v2_b'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=(kernel_size[0], kernel_size[1]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=(kernel_size[1], kernel_size[0]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x0_0 = layers.Concatenate()([x1_3, x2_1])
    x0_1 = conv2d_bn(x0_0, filters[3], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=None, name=name + 'x3_2')
    out = layers.Add()([x, x0_1])
    return out


def inception_resnet_v2_c(x, filters=(192, 224, 256, 2048), kernel_size=(1, 3), padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='inception_resnet_v2_c'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=(kernel_size[0], kernel_size[1]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=(kernel_size[1], kernel_size[0]), padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x0_0 = layers.Concatenate()([x1_3, x2_1])
    x0_1 = conv2d_bn(x0_0, filters[3], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=None, name=name + 'x3_2')
    out = layers.Add()([x, x0_1])
    return out


def reduction_resnet_v2_a(x, filters=(256, 384), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_resnet_v2_a'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[1], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x3_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(
        2, 2), padding=padding, name='pool')(x)
    out = layers.Concatenate()([x1_3, x2_1, x3_1])
    return out


def reduction_resnet_v2_b(x, filters=(256, 288, 320, 384), kernel_size=(1, 3), padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='reduction_resnet_v2_b'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[1], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[2], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    x2_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')
    x2_2 = conv2d_bn(x2_1, filters[1], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_2')
    x3_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_1')
    x3_2 = conv2d_bn(x3_1, filters[3], kernel_size=kernel_size[1], strides=2, padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3_2')
    x4_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(
        2, 2), padding=padding, name='pool')(x)
    out = layers.Concatenate()([x1_3, x2_2, x3_2, x4_1])
    return out


def restnet_block_a(x, filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='restnet_block_a'):
    x1 = conv2d_bn(x, filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1')
    x2 = conv2d_bn(x1, filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2')
    out = layers.Add()([x, x2])
    return out


def restnet_block_b(x, filters=(64, 256), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='restnet_block_b'):
    x1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1')
    x2 = conv2d_bn(x1, filters[0], kernel_size=kernel_size[1], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2')
    x3 = conv2d_bn(x2, filters[1], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3')
    out = layers.Add()([x, x3])
    return out


def resnext_block_a(x, filters=(4, 256), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='resnext_block_a'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')
    x1_3 = conv2d_bn(x1_2, filters[1], kernel_size=kernel_size[0], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_3')
    ...

    x32_1 = conv2d_bn(x1_3, filters[0], kernel_size=kernel_size[0], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x32_1')
    x32_2 = conv2d_bn(x32_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x32_2')
    x32_3 = conv2d_bn(x32_2, filters[1], kernel_size=kernel_size[0], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x32_3')
    x00 = layers.Concatenate()([x1_3, ..., x32_3])
    out = layers.Add()([x, x00])
    return out


def resnext_block_b(x, filters=(4, 256), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='resnext_block_b'):
    x1_1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_1')
    x1_2 = conv2d_bn(x1_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1_2')

    ...

    x32_1 = conv2d_bn(x1_2, filters[0], kernel_size=kernel_size[0], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x32_1')
    x32_2 = conv2d_bn(x32_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x32_2')
    x0 = layers.Concatenate()([x1_2, ..., x32_2])
    x00 = conv2d_bn(x0, filters[1], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x00')
    out = layers.Add()([x, x00])
    return out


def resnext_block_c(x, filters=(128, 256), kernel_size=(1, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(5e-4), activation='relu', name='resnext_block_c'):
    x1 = conv2d_bn(x, filters[0], kernel_size=kernel_size[0], padding=padding, kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x1')
    x2_1 = conv2d_bn(x1, filters[0], kernel_size=kernel_size[1], padding=padding,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_1')

    ...

    x2_32 = conv2d_bn(x2_1, filters[0], kernel_size=kernel_size[1], padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x2_32')
    x3 = conv2d_bn(x2_32, filters[1], kernel_size=kernel_size[0], padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer, activation=activation, name=name + 'x3')
    out = layers.Add()([x, x3])
    return out


def attention_module(input_tensor, kernel_size, stage, block_name, filters=(32, 64),
                     kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)):
    """Hourglass Attention Module.

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

                  | —————————————————————
                  ↓                     |
           resnet_identity_block        |
                  ↓                     ↓
           resnet_identity_block        |
                  ↓                     ↓
             conv2d BN relu     conv2d /2 BN relu
                  ↓                     ↓
               conv2d BN                |
                 (+) ———————————————————
                  ↓
                relu

       """
    filter1, filter2 = filters

    xres1 = resnet_identity_block(input_tensor, (filter1, filter1, filter2), kernel_size, stage, block_name + '_res1',
                                  kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    xres2 = resnet_identity_block(xres1, (filter2, filter2, filter2), kernel_size, stage, block_name + '_res2',
                                  kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    # encoder-decoder
    xed = layers.MaxPool2D()(xres1)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed1',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed1_1',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    xed = layers.MaxPool2D()(xed)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed2',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed2_1',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    xed = layers.Conv2DTranspose(
        filter2, kernel_size, strides=2, padding='same')(xed)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed3',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    xed = resnet_identity_block(xed, (filter2, filter2, filter2), kernel_size, stage, block_name + '_resed3_1',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

    xed = layers.Conv2DTranspose(
        filter1, kernel_size, strides=2, padding='same')(xed)
    xed = layers.Conv2D(filter2, kernel_size=(
        1, 1), padding='same', activation='sigmoid')(xed)
    msk = layers.Lambda(lambda x: 1 + x)(xed)

    xmul = layers.Multiply()([xres2, msk])
    xadd = layers.Add()([xres2, xmul])
    xadd = layers.Activation('relu')(xadd)
    xed = resnet_identity_block(xadd, (filter1, filter1, filter2), kernel_size, stage, block_name + '_resed',
                                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    return xed


def densenet_denseblock(x, stage, nb_layers=4, nb_filter=32, growth_rate=16, kernel_initializer='he_normal',
                        use_batchnorm=True, name='dsnt', kernel_regularizer=regularizers.l2(5e-4), conv_simple_type=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
       Output
          ↑
    aaaaaaaaaaaaaa
    bbbbbbbbbbbbbb
    cccccccccccccc
    dddddddddddddd
          ↑
    aaaaaaaaaaaaaa
    bbbbbbbbbbbbbb
    cccccccccccccc
          ↑
    aaaaaaaaaaaaaa
    bbbbbbbbbbbbbb
          ↑
    aaaaaaaaaaaaaa
          ↑
        input

    '''
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        with tf.name_scope(name + 'ds' + str(stage)):
            if conv_simple_type:
                x = conv2d_bn(input_tensor=concat_feat, filters=growth_rate, kernel_size=(3, 3), use_batchnorm=use_batchnorm,
                              name=name + 'dsb_' + str(branch), kernel_initializer=kernel_initializer, activation='relu',
                              kernel_regularizer=kernel_regularizer)
            else:
                x = resnet_shortcut_block(input_tensor=concat_feat, filters=[growth_rate, growth_rate],
                                          strides=(1, 1), use_batchnorm=use_batchnorm,
                                          kernel_size=(3, 3), stage=stage, block_name=name + 'dsb_' + str(branch),
                                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
            concat_feat = layers.Concatenate()([concat_feat, x])
        nb_filter += growth_rate
    return concat_feat, nb_filter


def densenet_transitionblock(x, stage, nb_filter=64, compression=1.0, dropout_rate=None, kernel_initializer='he_normal',
                             use_batchnorm=True, name='dts', kernel_regularizer=regularizers.l2(5e-4)):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = name + 'ts' + str(stage) + '_blk'
    relu_name_base = name + 'tsat' + str(stage) + '_blk'

    x = layers.Conv2D(int(nb_filter * compression), kernel_size=1, strides=1, kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer, name=conv_name_base, use_bias=False)(x)
    if use_batchnorm:
        x = layers.BatchNormalization(
            epsilon=eps, name=conv_name_base + '_bn')(x)
    x = layers.Activation('relu', name=relu_name_base)(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x


def densenet_module(input_tensor, densblock_number=3, nlayer_in_block=4, growth_rate=16, compression=1.0, stage=0,
                    dropout_rate=None, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001),
                    use_batchnorm=True, name='dsmd', conv_simple_type=False):
    x = input_tensor
    for i in range(densblock_number):
        x, nb_filter = densenet_denseblock(x, stage=i + stage, nb_layers=nlayer_in_block, growth_rate=growth_rate,
                                           name=name + '_dbk' + str(i), use_batchnorm=use_batchnorm, conv_simple_type=conv_simple_type,
                                           kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        x = densenet_transitionblock(x, stage=i + stage, nb_filter=nb_filter, compression=compression,
                                     name=name + '_ts' + str(i), use_batchnorm=use_batchnorm,
                                     dropout_rate=dropout_rate,
                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    return x


if __name__ == '__main__':
    # example DensNet
    with tf.device('/cpu:0'):
        inputs = layers.Input((512, 512, 1))
        out = densenet_module(inputs, densblock_number=4,
                              nlayer_in_block=6, growth_rate=16, compression=0.5)
        model = models.Model(inputs, out, name='example_cnn')
        model.summary()
        model.save('dens.h5')
