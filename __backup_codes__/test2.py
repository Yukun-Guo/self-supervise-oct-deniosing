import keras
from keras import layers as L
class my_model(keras.Model):
    def __init__(self, dim):
        super(my_model, self).__init__()
        self.Base  = keras.applications.VGG16(
            input_shape=(dim), 
            include_top = False, 
            weights = 'imagenet'
        )
        self.GAP   = L.GlobalAveragePooling2D()
        self.BAT   = L.BatchNormalization()
        self.DROP  = L.Dropout(rate=0.1)
        self.DENS  = L.Dense(256, activation='relu', name = 'dense_A')
        self.OUT   = L.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x  = self.Base(inputs)
        g  = self.GAP(x)
        b  = self.BAT(g)
        d  = self.DROP(b)
        d  = self.DENS(d)
        return self.OUT(d)
    
    # AFAIK: The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        x = L.Input(shape=dim)
        return keras.Model(inputs=[x], outputs=self.call(x))

dim = (124,124,3)
model = my_model((dim))
model.build((None, *dim))
model.build_graph().summary()
keras.utils.plot_model(model.build_graph(), show_shapes=True)