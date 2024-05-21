import os
import cv2
from template_cnn import UNet2D
from utils import utils
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# save image callback
def save_images_callback(save_path):
    def callback(data):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # get file name
        fn = os.path.basename(data['paths'])
        # genereate out file name
        out_fn = os.path.join(save_path, fn.replace('.png','_out.png'))
        cv2.imwrite(out_fn, data['out'])          
    return callback

# parameters
nclass = 12
batch_size = 1
n_epoch = 900
data_shape = (480, 320,1)

# create dataset
bscan_list = utils.listFiles('./data/BscanOCT1','*.png') #BscanOCT1 images11 \test

# create model
model = UNet2D(in_size=(None,None),in_channels=1,out_channels=1, restore_model=True)

model.load_model()

model.predict(bscan_list, callbacks=[save_images_callback('./logs/prediction')])
