import tensorflow as tf
import numpy as np
import PIL.Image as Image

# read image file
def tf_load_image():
    def _read_image(fn, color_mode=None):
        """
        Read image from file.
        :param image_file_name: full file path
        :param color_mode: 'gray', 'rgb' or 'idx'
        :return: numpy image with shape (height, width, channels)
        """
        img = Image.open(fn.numpy().rstrip())
        if color_mode is not None:
            color_mode = color_mode.numpy()
            if color_mode.lower() == 'gray':
                img = img.convert('L')
            else:
                if color_mode.lower() == 'rgb':
                    img = img.convert('RGB')
                else:
                    if color_mode.lower() == 'idx':
                        img = img.convert('P')
        return np.array(img)
    
    def _tf_read_image(image_file_name, color_model='idx'):
        img = tf.py_function(_read_image, [image_file_name, color_model], [tf.float32])
        return tf.transpose(img, perm=[1, 2, 0])
    
    def wrapper(filename):
        out_image = _tf_read_image(filename, 'gray')
        # generate label file name
        filename = tf.strings.regex_replace(filename, 'Bscans_temp', 'Labels_temp')
        filename = tf.strings.regex_replace(filename, '_1_bscan_', '_label_')
        out_label = _tf_read_image(filename, 'gray')
        return out_image, out_label
    return wrapper

# random brightness
def tf_random_brightness(max_delta=0.2):
    def wrapper(image, label):
        image = tf.image.stateless_random_brightness(image, max_delta,np.random.randint(10,size=2))
        return image, label
    return wrapper

# random crop image
def tf_random_crop(out_shape,data_pad_value=0):
    def wrapper(img, label):
        # label_exp = tf.expand_dims(label, 2)
        cated = tf.concat([img, label], axis=-1)
        sz = tf.shape(cated)[:2]
        out_sz = tf.convert_to_tensor(out_shape)
        rg = tf.subtract(sz,out_sz)
        # padding if needed
        pad_value = tf.transpose(tf.stack([tf.constant([0, 0]), -tf.minimum(rg, 0)]))
        pad_value = tf.concat([pad_value, tf.constant([[0, 0]])], axis=0)
        cated = tf.pad(cated, paddings=pad_value, constant_values=data_pad_value)
        # random crop
        out_sz = tf.concat([out_sz, tf.constant([2])], axis=0)
        crped = tf.image.stateless_random_crop(cated, out_sz ,np.random.randint(10,size=2))
        
        #split image and label
        img = tf.expand_dims(crped[:, :, 0], 2)
        lbl = tf.expand_dims(crped[:, :, 1], 2)
        return img, lbl
    return wrapper

# add noise : gaussian and poisson
def tf_add_noise(mean=20, std=6):
    def wrapper(image, label):
        # generate random mean and std       
        mean_rand = tf.random.uniform((), -mean, mean/2)
        std_rand = tf.random.uniform((), 0, std)
        peak = tf.random.uniform((), 0.5, 0.85)
        
        gaussian_noise = tf.random.normal(tf.shape(image), mean=mean_rand, stddev=std_rand)
        poisson_noise = tf.random.poisson(tf.shape(image),peak)*peak*50
        image_noise = image + gaussian_noise*1.3 + poisson_noise
        image_noise = tf.clip_by_value(image_noise, 0., 255.)
        return image_noise, label
    return wrapper

# random rotate90
def tf_random_rotate90n():
    def wrapper(image, label):
        degree = np.random.randint(0, 3)
        image = tf.image.rot90(image, degree)
        label = tf.image.rot90(label, degree)
        return image, label
    return wrapper

# random horizontal flip
def tf_random_horizontal_flip():
    def wrapper(image, label):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        return image, label
    return wrapper

# random vertical flip
def tf_random_vertical_flip():
    def wrapper(image, label):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
        return image, label
    return wrapper

# normalize
def tf_normalize():
    def wrapper(image, label):
        image = image / 255.0
        label = label / 255.0
        return image, label
    return wrapper

# to tensor
def tf_to_tensor(nclass=12):
    def wrapper(image, label):
        # set shape
        image.set_shape((None,None,1))
        label.set_shape((None,None,1))
        return image, label 
    return wrapper

def MyPyDatasetTF(imgs,data_size=(300,200),n_class=12, batch_size=2, use_data_augmentation=True):

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(imgs)))
    dataset = dataset.map(tf_load_image())
    if use_data_augmentation:
        # dataset = dataset.map(tf_random_brightness())
        dataset = dataset.map(tf_random_crop(data_size))
        dataset = dataset.map(tf_add_noise(mean=40, std=40))
        # dataset = dataset.map(tf_random_rotate90n())
        dataset = dataset.map(tf_random_horizontal_flip())
        # dataset = dataset.map(tf_random_vertical_flip())
    dataset = dataset.map(tf_normalize())
    dataset = dataset.map(tf_to_tensor(n_class))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':  
    
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import utils.utils as utils
    
    imgs = utils.listFiles('./data/images','*.png')
    
    # create dataset
    my_dataset = MyPyDatasetTF(imgs,data_size=(320,256),n_class=1, batch_size=1, use_data_augmentation=True)
    
    # test dataset
    for img,lbl in my_dataset:
        img_np = img.numpy().squeeze(0)
        lbl_np = lbl.numpy().squeeze(0)
        
        print(img_np.shape)
        print(lbl_np.shape)
        
        

        
