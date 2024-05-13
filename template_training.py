"""
resumable training version 0.1
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from template_dataset import MyPyDataset
from template_cnn import ExampleCNN
from utils import utils


# parameters
nclass = 12
batch_size = 4
n_epoch = 500 
data_shape = (480, 288,1)
# create dataset
bscan_list = utils.listFiles('./data/images','*.png')
groundtruth_list = utils.listFiles('./data/groundtruth','*.png')
bscan_train_list, bscan_test_list, bscan_valid_list = utils.split_list(bscan_list,split=(0.8,0,0.2))
gt_train_list, gt_test_list, gt_valid_list = utils.split_list(groundtruth_list,split=(0.8,0,0.2))

train_dataset = MyPyDataset(imgs=bscan_train_list, lbls=gt_train_list, data_size=data_shape[0:2],n_class=nclass,batch_size=batch_size)
validation_dataset = MyPyDataset(imgs=bscan_valid_list, lbls=gt_valid_list, data_size=data_shape[0:2],n_class=nclass,batch_size=batch_size)

# create model 
my_model = ExampleCNN(input_shape=data_shape, n_class=nclass,restore_model=True)  

my_model.train(train_dataset,validation_data=validation_dataset,batch_size=batch_size,epochs=n_epoch)
