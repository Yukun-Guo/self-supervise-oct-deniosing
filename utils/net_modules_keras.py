"""
utility blocks for network
Version 2.0
Yukun Guo
CEI, OHSU
"""
from keras import layers, regularizers

class Conv2DBN(object):
    def __init__(self, filters,
                 kernel_size,
                 padding='same',
                 strides=(1, 1),
                 dilation_rate=(1, 1),
                 use_bias=False,
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(0.001),
                 use_batchnorm=True,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.cov2d = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,**kwargs)
        self.bn = layers.BatchNormalization()
        self.atv = layers.Activation(activation)
        
    def __call__(self, inputs):
        xi = self.cov2d(inputs)
        if self.use_batchnorm:
            xi = self.bn(xi)
        if self.activation is not None:
            xi = self.atv(xi)
        return xi    

class ResnetIdentityBlock(object):
    def __init__(self, filters, kernel_size, kernel_initializer='he_normal',use_batchnorm=True,
                 kernel_regularizer=regularizers.l2(0.001), **kwargs):
        self.conv2d_bn1 = Conv2DBN(filters, kernel_size, padding='same', activation='relu',use_batchnorm=use_batchnorm,
                                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        self.conv2d_bn2 = Conv2DBN(filters, kernel_size, padding='same', activation=None,use_batchnorm=use_batchnorm,
                                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        self.relu = layers.Activation('relu')
    def __call__(self, inputs):
        xi = self.conv2d_bn1(inputs)
        xi = self.conv2d_bn2(xi)
        xi = layers.add([xi, inputs])
        xi = self.relu(xi)
        return xi

class ResnetShortcutBlock(object):
    def __init__(self, filters, kernel_size, strides=(2, 2), use_batchnorm=True,
                 kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), **kwargs):
        # check type of filters, if filters is int change it to list
        if isinstance(filters, int):
            filters = [filters, filters]
        self.conv2d_bn1 = Conv2DBN(filters[0], kernel_size, strides=strides, activation='relu',
                                   kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm,
                                   kernel_initializer=kernel_initializer)
        self.conv2d_bn2 = Conv2DBN(filters[1], kernel_size, padding='same', activation=None, kernel_regularizer=kernel_regularizer,
                                   kernel_initializer=kernel_initializer, use_batchnorm=use_batchnorm)
        self.conv2d_bn3 = Conv2DBN(filters[1], (1, 1), strides=strides, activation=None, use_batchnorm=use_batchnorm,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_initializer=kernel_initializer)
        self.relu = layers.Activation('relu')
    def __call__(self, inputs):
        xi = self.conv2d_bn1(inputs)
        xi = self.conv2d_bn2(xi)
        shortcut = self.conv2d_bn3(inputs)
        xi = layers.add([xi, shortcut])
        xi = self.relu(xi)
        return xi

class Conv3DBN(object):
    def __init__(self, filters, kernel_size, padding='same', strides=(1, 1, 1), dilation_rate=(1, 1, 1),
                 kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), use_batchnorm=True, activation=None):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.cov3d = layers.Conv3D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization()
        self.atv = layers.Activation(activation)

    def __call__(self, inputs):
        xi = self.cov3d(inputs)
        if self.use_batchnorm:
            xi = self.bn(xi)
        if self.activation is not None:
            xi = self.atv(xi)
        return xi

class ResnetIdentityBlock3D(object):
    def __init__(self, filters, kernel_size, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(0.001), **kwargs):
        self.conv3d_bn1 = Conv3DBN(filters, kernel_size, padding='same', activation='relu',
                                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        self.conv3d_bn2 = Conv3DBN(filters, kernel_size, padding='same', activation=None,
                                   kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
        self.relu = layers.Activation('relu')
    def __call__(self, inputs):
        xi = self.conv3d_bn1(inputs)
        xi = self.conv3d_bn2(xi)
        xi = layers.add([xi, inputs])
        xi = self.relu(xi)
        return xi

class ResnetShortcutBlock3D(object):
    def __init__(self, filters, kernel_size, strides=(2, 2, 2), use_batchnorm=True,
                 kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), **kwargs):
        # check type of filters, if filters is int change it to list
        if isinstance(filters, int):
            filters = [filters, filters]
        self.conv3d_bn1 = Conv3DBN(filters[0], kernel_size, strides=strides, activation='relu',
                                   kernel_regularizer=kernel_regularizer, use_batchnorm=use_batchnorm,
                                   kernel_initializer=kernel_initializer)
        self.conv3d_bn2 = Conv3DBN(filters[1], kernel_size, padding='same', activation=None, kernel_regularizer=kernel_regularizer,
                                   kernel_initializer=kernel_initializer, use_batchnorm=use_batchnorm)
        self.conv3d_bn3 = Conv3DBN(filters[1], (1,1,1), strides=strides, activation=None, use_batchnorm=use_batchnorm,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_initializer=kernel_initializer)
        self.relu = layers.Activation('relu')
    def __call__(self, inputs):
        xi = self.conv3d_bn1(inputs)
        xi = self.conv3d_bn2(xi)
        shortcut = self.conv3d_bn3(inputs)
        xi = layers.add([xi, shortcut])
        xi = self.relu(xi)
        return xi

if __name__ == '__main__':
    pass
