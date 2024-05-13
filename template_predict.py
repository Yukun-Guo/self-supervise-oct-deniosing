import os
import cv2
from template_cnn import ExampleCNN
from utils import utils
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# save image callback
def save_images_callback(save_path):
    def callback(data):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(len(data['paths'])):
            # get data[i]['paths'] base file name
            fn = os.path.basename(data['paths'][i])
            # genereate msks file name
            msk_fn = os.path.join(save_path, fn.replace('.png','_msk.png'))
            # genereate out file name
            out_fn = os.path.join(save_path, fn.replace('.png','_out.png'))
            
            # apply label color map
            msk = utils.apply_colormap(data['msk'][i].astype('uint8'))
            out = utils.apply_colormap(data['out'][i].astype('uint8'))
            
            # save images
            cv2.imwrite(msk_fn, msk)
            cv2.imwrite(out_fn, out)          
    return callback

# parameters
nclass = 12
batch_size = 1
n_epoch = 900
data_shape = (480, 320,1)

# create dataset
bscan_list = utils.listFiles('./data/images','*.png')

# create model
model = ExampleCNN(input_shape=(None,None,1), n_class=nclass)

model.load_model()

model.predict(bscan_list, batch_size=batch_size, callbacks=[save_images_callback('./logs/prediction')])
