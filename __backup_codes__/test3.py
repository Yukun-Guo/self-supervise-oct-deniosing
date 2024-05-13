import keras
from keras import layers, regularizers

def Conv2DBN(filters, 
             kernel_size, 
             padding='same', 
             strides=(1, 1), 
             dilation_rate=(1, 1), 
             use_bias=False,
             kernel_initializer='he_normal', 
             kernel_regularizer=regularizers.l2(0.001),
             batch_norm=True, activation=None):
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
    # define layers
    cov2d = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name='conv2d')
    bn = layers.BatchNormalization(scale=False)
    atv = layers.Activation(activation)
            
    def forward(input_tensor):
        xi = cov2d(input_tensor)
        if batch_norm:
            xi = bn(xi)
        if activation is not None:
            xi = atv(xi)
        return xi
    return forward

# convert Conv2DBN to a keras layer
class Conv2DBNLayer(layers.Layer):
    def __init__(self, filters, 
             kernel_size, 
             padding='same', 
             strides=(1, 1), 
             dilation_rate=(1, 1), 
             use_bias=False,
             kernel_initializer='he_normal', 
             kernel_regularizer=regularizers.l2(0.001),
             batch_norm=True, activation=None):
        super(Conv2DBNLayer, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.cov2d = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name='conv2d')
        self.bn = layers.BatchNormalization()
        self.atv = layers.Activation(activation)
    
    def call(self, inputs):
        xi = self.cov2d(inputs)
        if self.batch_norm:
            xi = self.bn(xi)
        if self.activation is not None:
            xi = self.atv(xi)
        return xi


# class my_model(keras.Model):
#     def __init__(self):
#         super(my_model, self).__init__()
#         self.conv = Conv2DBNLayer(64, kernel_size=3, padding="same", activation="relu")
    
#     def call(self, inputs):
#         x  = self.conv(inputs)
#         return x
    
#     # AFAIK: The most convenient method to print model.summary() 
#     # similar to the sequential or functional API like.
#     def build_graph(self):
#         x = layers.Input(shape=dim)
#         return keras.Model(inputs=[x], outputs=self.call(x))
# dim = (124,124,3)
# model = my_model((dim))
# model.build((None, *dim))
# model.build_graph().summary()
# keras.utils.plot_model(model.build_graph(), show_shapes=True)
# model.build_graph().save("my_model.keras")

################################################################################################
# # build sequential model
# my_model = keras.Sequential()
# my_model.add(layers.Input(shape=(124, 124, 3)))
# my_model.add(layers.Conv2D(16, kernel_size=3, padding="same"))
# my_model.add(Conv2DBNLayer(64, kernel_size=3, padding="same", activation="relu"))
# # my_model.build((None, 124, 124, 3))
# my_model.summary()

################################################################################################
# buld functioal model
# build model
def my_model():
    inputs = keras.Input(shape=(124, 124, 3))
    x = layers.Conv2D(16, kernel_size=3, padding="same")(inputs)
    x = Conv2DBNLayer(64, kernel_size=3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(10, kernel_size=3, padding="same")(x)
    return keras.Model(inputs, outputs)

model = my_model()
model.summary()
keras.utils.plot_model(model, show_shapes=True)
model.save("my_model.keras")
