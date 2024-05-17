"""
resumable training version 0.1
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tensorflow as tf
from template_dataset_tf import MyPyDatasetTF
from template_cnn import UNet2D
from utils import utils

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# parameters
nclass = 1
batch_size = 4
n_epoch = 500 
data_size = (480, 288)
# create dataset
bscan_list = utils.listFiles('./data/images','*.png')

bscan_train_list, bscan_test_list, bscan_valid_list = utils.split_list(bscan_list,split=(0.8,0,0.2))

train_dataset = MyPyDatasetTF(imgs=bscan_train_list, data_size=data_size[0:2],n_class=nclass,batch_size=batch_size)
validation_dataset = MyPyDatasetTF(imgs=bscan_valid_list, data_size=data_size[0:2],n_class=nclass,batch_size=batch_size)

# create model 
# with strategy.scope():
my_model = UNet2D(in_size=data_size,in_channels=1,out_channels=1, restore_model=False)  

my_model.train(train_dataset,validation_data=validation_dataset,batch_size=batch_size,epochs=n_epoch)
