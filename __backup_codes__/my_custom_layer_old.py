import numpy as np
import keras
from keras import layers, ops

class Linear(keras.layers.Layer):
    '''
        y = w.x + b
        w is a trainable parameter
        b is a trainable parameter
    '''
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    # If you need your custom layer to support serialization, you can optionally implement a get_config method:
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
class ComputeSum(keras.layers.Layer):
    '''
        y = sum(x)
        x is not a trainable parameter
    '''
    def __init__(self, input_dim):
        super().__init__()
        self.total = self.add_weight(
            initializer="zeros", shape=(input_dim,), trainable=False
        )

    def call(self, inputs):
        self.total.assign_add(ops.sum(inputs, axis=0))
        return self.total
    
class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed_generator = keras.utils.set_random_seed(42)
    # training is a boolean indicating whether the layer should behave in training mode or in inference mode
    def call(self, inputs, training=None): 
        if training:
            return keras.random.dropout(inputs, self.rate, seed=self.seed_generator)
        return inputs