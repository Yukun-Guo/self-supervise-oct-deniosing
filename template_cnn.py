import os
import glob
import keras
import math
import numpy as np
from keras import layers
from keras.layers import BatchNormalization, MaxPool2D,UpSampling2D,Conv2D
from utils.net_modules_keras import Conv2DBN, ResnetIdentityBlock, ResnetShortcutBlock
from utils import utils, losses, metrics

def get_encoder(filters):
    xs = []
    def wrapper(input_tensor,mask=None):
        for i in range(0,len(filters)):
            if i == 0:
                x = Conv2DBN(filters[0],kernel_size=(7,7),use_batchnorm=True,activation='relu')(input_tensor)
                if mask is not None:
                    x = keras.ops.multiply(keras.ops.add(x,keras.ops.convert_to_tensor([1,])), mask)
                xs.append(x)
            else:
                x = ResnetShortcutBlock(filters[i],kernel_size=(3,3))(x)
                x = ResnetIdentityBlock(filters[i],kernel_size=(3,3))(x)
                x = ResnetIdentityBlock(filters[i],kernel_size=(3,3))(x)
                if mask is not None:
                    mask = MaxPool2D()(mask)
                    x = keras.ops.multiply(keras.ops.add(x,keras.ops.convert_to_tensor([1,])), mask)
                xs.append(x)
        return xs
    return  wrapper

def get_decoder(filters,output_channels=1,output_activation='sigmoid',out_name='out'):
    def wrapper(xs):
        for i in range(len(filters)-1,-1,-1):
            if i == len(filters)-1:
                x = ResnetShortcutBlock(filters[i],strides=(1,1),kernel_size=(3,3))(xs[i])
                x = ResnetIdentityBlock(filters[i],kernel_size=(3,3))(x)
                x = layers.concatenate([x,xs[i]],axis=-1)
                x = ResnetShortcutBlock(filters[i],strides=(1,1),kernel_size=(3,3))(x)
                x = ResnetIdentityBlock(filters[i],kernel_size=(3,3))(x)
                x = UpSampling2D()(x)
            else:
                x = layers.concatenate([x,xs[i]],axis=-1)
                x = ResnetShortcutBlock(filters[i],strides=(1,1),kernel_size=(3,3))(x)
                x = ResnetIdentityBlock(filters[i],kernel_size=(3,3))(x)
                if i != 0:
                    x = UpSampling2D()(x)
        x = Conv2DBN(filters[0],kernel_size=(3,3),use_batchnorm=True,activation='relu')(x)
        x = Conv2D(output_channels,kernel_size=(3,3),padding='same',activation=output_activation,name=out_name)(x)
        return x
    return wrapper             
        
class ExampleCNN(object):
    
    def __init__(self,input_shape, n_class=13, restore_model=False):
        super().__init__()
        self.input_shape = input_shape
        self.n_class = n_class
        self.downsample_level = 5
        self.checkpoint_dir='./logs'
        self.log_dir='./logs'
        self.model_name = "my_example_model"
        
        # parameters for compile
        self.optimizer = self._config_optimizer()
        self.loss = self._config_loss()
        self.loss_weights=None
        self.metrics = self._config_metrics()
        self.weighted_metrics=None
        self.run_eagerly=False
        self.steps_per_execution=1
        self.jit_compile="auto"
        self.auto_scale_loss=True
        
        # parameters for train
        self.verbose="auto"
        self.callbacks=self._configure_callbacks()
        self.shuffle=True
        self.class_weight=None
        self.sample_weight=None
        self.initial_epoch=0
        self.steps_per_epoch=None
        self.validation_steps=None
        self.validation_batch_size=None
        self.validation_freq=1
        
        self.model = self._build_or_restore_model(restore_model)


    def train(self, train_data,validation_split=0.0,validation_data=None,batch_size=None, epochs=5000):
        return self.model.fit(x=train_data,
                       y=None,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=self.verbose,
                       callbacks=self.callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data,
                       shuffle=self.shuffle,
                       class_weight=self.class_weight,
                       sample_weight=self.sample_weight,
                       initial_epoch=self.initial_epoch,
                       steps_per_epoch=self.steps_per_epoch,
                       validation_steps=self.validation_steps,
                       validation_batch_size=self.validation_batch_size,
                       validation_freq=self.validation_freq)
    
    def evaluate(self, x, y=None,batch_size=None,verbose="auto",sample_weight=None,steps=None,callbacks=None,return_dict=False,**kwargs):
        return self.model.evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,return_dict,*kwargs)
        
    def predict(self, x, batch_size=1, verbose="auto", callbacks=None):
        if isinstance(x,str):
            x = [x]
        x_in = []
        len_x = len(x)
        num_batch = math.ceil(len_x / batch_size)
        
        for i in range(num_batch):
            low = i * batch_size
            high = min(low + batch_size, len_x)
            x_batch = x[low:high]
            x_batch = [utils.read_image(x_path,'gray') for x_path in x_batch]
            raw_img_size = [img.shape[0:2] for img in x_batch]
            x_batch = [np.expand_dims(utils.resize_pow2_size(img, self.downsample_level),2) for img in x_batch]
            x_in = np.stack(x_batch, axis=0)/255.0
            msk,out = self.model.predict_on_batch(x_in)
            # msk = np.argmax(msk, axis=3)
            msk = np.where(msk>0.5,1,0)
            out = np.argmax(out, axis=3)
            # resize to raw image size
            msk = [utils.restore_resized_pow2size(img,raw_img_size[i]) for i,img in enumerate(msk)]
            out = [utils.restore_resized_pow2size(img,raw_img_size[i]) for i,img in enumerate(out)]
            msk = np.clip(msk,0,255).astype('uint8')
            out = np.clip(out,0,255).astype('uint8')

            # call backs
            if callbacks is not None:
                for callback in callbacks:
                    callback({'paths':x[low:high],'msk':msk,'out':out})
                    
            if verbose == "auto":
                print("predicting {}/{} batch".format(i+1,num_batch))
            elif verbose == 1:
                print("predicting {}/{} batch".format(i+1,num_batch))
        # return msk,out
    
    def load_model(self,model_path=None, load_best_model=True):
        try :
            if  model_path is not None:
                model = keras.models.load_model(model_path)
                self.input_shape=(None,None,1)
                self.model = self._build_model()
                self.model.set_weights(model.get_weights())
                return
            if load_best_model:
                checkpoints = list(glob.iglob(os.path.join( self.checkpoint_dir, '*.keras'), recursive=False))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print("Load from", latest_checkpoint)
                    model = keras.models.load_model(latest_checkpoint)
                    self.input_shape=(None,None,1)
                    self.model = self._build_model()
                    self.model.set_weights(model.get_weights())
                else:
                    raise Exception("No model found!")            
        except  Exception as error:
            print("Error: load model failed!",error)
    
    def plot(self):
        keras.utils.plot_model(self.model, show_shapes=True)
            
    def _build_or_restore_model(self, restore_model=True):
        if restore_model:
            checkpoints = list(glob.iglob(os.path.join( self.checkpoint_dir, '*.keras'), recursive=False))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print("Restoring from", latest_checkpoint)
                return keras.saving.load_model(latest_checkpoint)
            print("Creating a new model")
            return self._build_model()
        else:
            print("Creating a new model")
            return self._build_model()
    
    def _build_model(self):
        inputs = keras.Input(shape=self.input_shape,name='input')
        
        # mask_features = get_encoder([32,64,128,196])(inputs)
        # out = get_decoder([32,64,128,196],output_channels=self.n_class,output_activation='softmax',out_name='out')(mask_features)
        
        
        
        mask_features = get_encoder([32,64,128,196])(inputs)
        mask = get_decoder([32,64,128,196],output_channels=1,output_activation='sigmoid',out_name='mask')(mask_features)
        
        features = get_encoder([32,64,128,196,256])(inputs,mask)
        out = get_decoder([32,64,128,196,256],output_channels=self.n_class,output_activation='softmax',out_name="out")(features)
        
        model = keras.Model(inputs=inputs, outputs=[mask, out], name=self.model_name)
        
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      loss_weights=self.loss_weights,
                      metrics=self.metrics,
                      weighted_metrics=self.weighted_metrics,
                      run_eagerly=self.run_eagerly,
                      steps_per_execution=self.steps_per_execution,
                      jit_compile=self.jit_compile,
                      auto_scale_loss=self.auto_scale_loss)
        model.summary()
        return model

    def _config_loss(self):
        loss = [losses.CombinedLoss([losses.DiceLoss(),losses.BinaryCrossEntropyLoss()]),losses.CombinedLoss([losses.DiceLoss(),losses.CategoricalCrossEntropyLoss()])]
        return loss
    
    def _config_metrics(self):
        metric = [metrics.DiceScore(),metrics.DiceScore()]
        return metric
    
    def _config_optimizer(self):
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        return optimizer
    
    def _configure_callbacks(self):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_dir+'/'+self.model_name+'_best.keras', #l{epoch:02d}
                save_best_only=True,  # Only save a model if `val_loss` has improved.
                # save_weights_only=True,
                monitor='val_loss',
                verbose=1),
            keras.callbacks.TensorBoard(log_dir= self.log_dir),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
            keras.callbacks.CSVLogger( self.log_dir+'/training.log')
        ]
        return callbacks
    
if __name__ == "__main__":
    model = ExampleCNN((304, 304, 1), n_class=12)
    model.plot()
    model.model.save("template_model.keras")

