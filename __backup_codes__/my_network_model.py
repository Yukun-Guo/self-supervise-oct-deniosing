import os
os.environ["KERAS_BACKEND"] = "tensorflow" # "pytorch" or "jax"

import keras
from keras import layers
from utils.net_modules_tf import Conv2DBN,ResnetShortcutBlock,ResnetIdentityBlock

# functional creation``
def get_model():
    # input 
    inputs = keras.Input(shape=(784,), name="digits")
    
    # hidden layers
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    
    # output
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    
    # model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# sequential creation
def get_sequential_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(784,), name="digits"),
            layers.Dense(64, activation="relu", name="dense_1"),
            layers.Dense(64, activation="relu", name="dense_2"),
            layers.Dense(10, activation="softmax", name="predictions"),
        ]
    )
    return model

# subclassing creation
class MyModel(keras.Model):
    def __init__(self, in_channels, num_classes=10, out_activation="softmax"):
        super(MyModel, self).__init__(name="my_model")
        
        self.num_classes = num_classes
        hid_chns = [64, 96, 128, 192, 128, 96, 64, 32, 16]
        #bottom
        self.bn1 = layers.BatchNormalization()
        self.conv1 = Conv2DBN(in_channels, kernel_size=3, padding="same", activation="relu",name="conv1")
        
        # block 1
        self.resnet_block1 = ResnetShortcutBlock(hid_chns[0],kernel_size=3,stage=1,block_name="block_1")
        self.resnet_block2 = ResnetIdentityBlock(hid_chns[0],kernel_size=3,stage=1,block_name="block_2")
        self.resnet_block3 = ResnetIdentityBlock(hid_chns[0],kernel_size=3,stage=1,block_name="block_3")
        self.pool_mask1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        
        # block 2
        self.resnet_block4 = ResnetShortcutBlock(hid_chns[0],kernel_size=3,stage=2,block_name="block_4")
        self.resnet_block5 = ResnetIdentityBlock(hid_chns[1],kernel_size=3,stage=2,block_name="block_5")
        self.resnet_block6 = ResnetIdentityBlock(hid_chns[1],kernel_size=3,stage=2,block_name="block_6")
        self.resnet_block7 = ResnetIdentityBlock(hid_chns[1],kernel_size=3,stage=2,block_name="block_7")
        self.pool_mask2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        
        # block 3
        self.resnet_block8 = ResnetShortcutBlock(hid_chns[1],kernel_size=3,stage=3,block_name="block_8")
        self.resnet_block9 = ResnetIdentityBlock(hid_chns[2],kernel_size=3,stage=3,block_name="block_9")
        self.resnet_block10 = ResnetIdentityBlock(hid_chns[2],kernel_size=3,stage=3,block_name="block_10")
        self.resnet_block11 = ResnetIdentityBlock(hid_chns[2],kernel_size=3,stage=3,block_name="block_11")
        self.resnet_block12 = ResnetIdentityBlock(hid_chns[2],kernel_size=3,stage=3,block_name="block_12")
        self.resnet_block13 = ResnetIdentityBlock(hid_chns[2],kernel_size=3,stage=3,block_name="block_13")
        self.pool_mask3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        
        # mid
        self.resnet_block14 = ResnetShortcutBlock(hid_chns[2],kernel_size=3,stage=4,block_name="block_14")
        self.resnet_block15 = ResnetIdentityBlock(hid_chns[3],kernel_size=3,stage=4,block_name="block_15")
        self.resnet_block16 = ResnetIdentityBlock(hid_chns[3],kernel_size=3,stage=4,block_name="block_16")
        self.up1 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        # cat up1 and resnet_block13
        
        # decoder block 1
        self.conv2 = Conv2DBN(hid_chns[4], kernel_size=3, padding="same", activation="relu")
        self.conv3 = Conv2DBN(hid_chns[4], kernel_size=3, padding="same", activation="relu")
        self.up2 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        
        # decoder block 2
        self.conv4 = Conv2DBN(hid_chns[5], kernel_size=3, padding="same", activation="relu")
        self.conv5 = Conv2DBN(hid_chns[5], kernel_size=3, padding="same", activation="relu")
        self.up3 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        
        # decoder block 3
        self.conv6 = Conv2DBN(hid_chns[6], kernel_size=3, padding="same", activation="relu")
        self.conv7 = Conv2DBN(hid_chns[6], kernel_size=3, padding="same", activation="relu")
        self.up4 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        
        # decoder block 4
        self.conv8 = Conv2DBN(hid_chns[6], kernel_size=3, padding="same", activation="relu")
        self.conv9 = Conv2DBN(hid_chns[7], kernel_size=3, padding="same", activation="relu")
        
        # output
        self.conv10 = Conv2DBN(hid_chns[7], kernel_size=3, padding="same", activation="relu")
        self.conv11 = Conv2DBN(hid_chns[8], kernel_size=3, padding="same", activation="relu")
        
        self.out = layers.Conv2D(self.num_classes, kernel_size=3, padding="same", activation=out_activation)
           
    def call(self, inputs):
        inputss, mask = inputs
        x = self.bn1(inputss)
        x1 = self.conv1(x)
        if mask is not None:
            x1 = (x1+1)*mask
        
        # block 1
        x = self.resnet_block1(x1)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        if mask is not None:
            mask = self.pool_mask1(mask)
            x2 = (x2+1)*mask
        MyModel
        x = self.resnet_block4(x2)
        x = self.resnet_block5(x)
        x = self.resnet_block6(x)
        x = self.resnet_block7(x)
        if mask is not None:
            mask = self.pool_mask2(mask)
            x3 = (x3+1)*mask
        
        # block 3
        x = self.resnet_block8(x3)
        x = self.resnet_block9(x)
        x = self.resnet_block10(x)
        x = self.resnet_block11(x)
        x = self.resnet_block12(x)
        x = self.resnet_block13(x)
        if mask is not None:
            mask = self.pool_mask3(mask)
            x4 = (x4+1)*mask
            
        # mid
        x = self.resnet_block14(x4)
        x = self.resnet_block15(x)
        x = self.resnet_block16(x)
        x = self.up1(x)
        x5 = layers.concatenate([x, x4], axis=-1)
        
        # decoder block 1
        x = self.conv2(x5)
        x = self.conv3(x)
        x = self.up2(x)
        x6 = layers.concatenate([x, x3], axis=-1)
        
        # decoder block 2
        x = self.conv4(x6)
        x = self.conv5(x)
        x = self.up3(x)
        x7 = layers.concatenate([x, x2], axis=-1)
        
        # decoder block 3
        x = self.conv6(x7)
        x = self.conv7(x)
        x = self.up4(x)
        x8 = layers.concatenate([x, x1], axis=-1)
        
        # decoder block 4
        x = self.conv8(x8)
        x = self.conv9(x)

        # output
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.out(x)
        
        return x

    def build_graph(self):
        x = layers.Input(shape=(512, 512, 1))
        return keras.Model(inputs=[x], outputs=self.call((x,None)))
    
class MyModel_test(keras.Model):
    def __init__(self, in_channels, num_classes=10, out_activation="softmax"):
        super(MyModel_test, self).__init__(name="my_model")
        
        self.num_classes = num_classes
        hid_chns = [64, 96, 128, 192, 128, 96, 64, 32, 16]
        #bottom
        self.bn1 = layers.BatchNormalization()
        self.conv1 = Conv2DBN(hid_chns[0], kernel_size=3, padding="same", activation="relu")
        
        # block 1
        # self.resnet_block1 = ResnetShortcutBlock(hid_chns[0],kernel_size=3,stage=1,block_name="block_1")
        
    def call(self, inputs):
        inputss, mask = inputs
        x = self.bn1(inputss)
        x1 = self.conv1(x)
        # if mask is not None:
        #     x1 = (x1+1)*mask
        
        # # block 1
        # x = self.resnet_block1(x1)
        
        return x1

    def build_graph(self):
        x = layers.Input(shape=(512, 512, 1))
        return keras.Model(inputs=[x], outputs=self.call((x,None)))
    



if __name__ == "__main__":
    input_shape = (512, 512, 1)
    model = MyModel_test(1,10)
    # model.summary()
    model.build((None, *input_shape))
    model.build_graph().summary()
    keras.utils.plot_model(model.build_graph(), show_shapes=True)

    # x_train = ...
    # y_train = ...    
    # model = MyModel1(1,10)
    
    # optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # model.compile(optimizer, loss=keras.losses.MeanSquaredError())
    # model.summary()
    # model.fit(x_train, x_train, epochs=200, batch_size=64)
