import os
import keras

# ###################################### train header ##################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # The GPU id to use, usually either "0" or "1";

keras.mixed_precision.set_global_policy('mixed_float16')
print('Mixed precision enabled: Compute dtype: {}, Variable dtype: {}'.format(keras.mixed_precision.global_policy().compute_dtype,
                                                                              keras.mixed_precision.global_policy().variable_dtype))
# ##################################### end train header ###############################################################

import tensorflow as tf
from keras import callbacks
from cnn import cnn, metrics_psnr, metrics_ssim
from cnn_2 import cnn_2, total_loss
from data_set import create_dataset_from_imgs, read_file_list, shuffle_lists
from my_callbacks import MyTensorBoard, MyModelCheckpoint, MyCSVLogger
from __backup_codes__.my_model import my_load_model
from __backup_codes__.my_model_tools import my_model_tools

# parameters
batch_size = 256

n_class = 3
epochs = 5000
image_size = (64, 64)  # (340, 64)

# create dataset
training_bscan = read_file_list(r'./data/train_bscan.txt')
training_label = read_file_list(r'./data/train_label.txt')
test_bscan = read_file_list(r'./data/test_bscan.txt')
test_label = read_file_list(r'./data/test_label.txt')

training_bscan, training_label = shuffle_lists(training_bscan, training_label)
test_bscan, test_label = shuffle_lists(test_bscan, test_label)

# create data generator
training_set = create_dataset_from_imgs(training_bscan, training_label, batch_size=batch_size, image_size=image_size)
test_set = create_dataset_from_imgs(test_bscan, test_label, batch_size=batch_size, image_size=image_size)

# callbacks
tensorboard_visualization = MyTensorBoard(log_dir=r'.\logs', write_graph=True, histogram_freq=1,
                                          write_layer_outputs='outputs', validation_data=test_set)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-8)
csv_logger = MyCSVLogger(r'.\logs', append=True)
model_checkpoint = MyModelCheckpoint(r'.\logs\model_{epoch:03d}_{val_loss:.3f}.hdf5', monitor='val_loss',
                                     save_best_only=True, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)
callbacks = [model_checkpoint, reduce_lr, csv_logger, early_stopping, tensorboard_visualization]

# training
model_cnn = cnn_2((*image_size, 1))
[model_cnn, init_epoch] = my_load_model(model_cnn, logdir=r'.\logs', checkpoint_file='checkpoint.ckp',
                                        custom_objects={'metrics_psnr': metrics_psnr, 'metrics_ssim': metrics_ssim,
                                                        total_loss: 'total_loss'})
model_cnn.fit(training_set,
              epochs=epochs,
              steps_per_epoch=100,
              initial_epoch=init_epoch,
              verbose=1,
              callbacks=callbacks,
              validation_data=test_set,
              validation_steps=50)
