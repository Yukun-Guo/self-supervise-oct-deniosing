import keras
import math
import numpy as np
from utils import utils

class MyPyDataset(keras.utils.PyDataset):

    def __init__(self, imgs, lbls, data_size,n_class,batch_size, use_data_augmentation=True, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data_size = data_size
        self.n_class = n_class
        samples = []
        for img,lbl in zip(imgs,lbls):
            samples.append({'img':img,'msk':lbl})
        self.samples = read_data()(samples)
        
        if use_data_augmentation:
            self.map = DataTransforms([#  random_rotate90n(),
                                        random_horizontal_flip(),
                                        #  random_vertical_flip(),
                                        random_brightness(),
                                        random_crop(self.data_size), 
                                        add_gaussian_noise(),
                                        normalize(),
                                        to_tensor(self.n_class)])
        else:
            self.map = DataTransforms([normalize(),to_tensor(self.n_class)])
    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.samples) / self.batch_size)
        # return len(self.samples)
    
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.samples))
        batch_sample = self.samples[low:high]
        return self.map(batch_sample)
        # return self.map(self.samples[idx])


class DataTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
                   

def read_data():
    def wrapper(samples):
        out_samples = []
        if type(samples) is not list:
            samples = [samples]
        for sample in samples:
            img = utils.read_image(sample['img'],'gray')
            msk2 = utils.read_image(sample['msk'],'idx')
            msk1 = np.clip(msk2, 0, 1)
            out_samples.append({'img':img,'mask1':msk1,'mask2':msk2})
        return out_samples
    return wrapper


def gray_jitter(bright_range=(-20, 20), contrast_range=(0.7, 1.3), max_value=255):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            bright_scale = np.random.uniform(bright_range[0], bright_range[1])
            contrast_scale = np.random.uniform(contrast_range[0], contrast_range[1])
            meanv = np.mean(img)
            img = (img - meanv) * contrast_scale + meanv
            img = img + bright_scale
            img = np.clip(img, 0, max_value)
            out_samples.append({'img':img,'mask1':mask1,'mask2':mask2})
        return out_samples
    return wrapper


def random_brightness(max_delta=0.2):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            img = img + np.random.uniform(-max_delta, max_delta)
            img = np.clip(img, 0, 255)
            out_samples.append({'img':img,'mask1':mask1,'mask2':mask2})
        return out_samples
    return wrapper
    
def add_gaussian_noise(mean=0., std=5.):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            img = np.clip(img + np.random.randn(*img.shape) * std + mean, 0, 255)
            out_samples.append({'img':img,'mask1':mask1,'mask2':mask2})
        return out_samples
    return wrapper


def random_rotate90n():
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            degree = np.random.randint(0, 3)
            img = np.rot90(img, degree)
            mask1 = np.rot90(mask1, degree)
            mask2 = np.rot90(mask2, degree)
            out_samples.append({'img': img, 'mask1': mask1, 'mask2': mask2})
            return out_samples
    return wrapper


def random_horizontal_flip(p=0.5):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            if np.random.random() < p:
                img = np.fliplr(img)
                mask1 = np.fliplr(mask1)
                mask2 = np.fliplr(mask2)
            out_samples.append({'img': img, 'mask1': mask1, 'mask2': mask2})
        return out_samples
    return wrapper


def random_vertical_flip(p=0.5):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            if np.random.random() < p:
                img = np.flipud(img)
                mask1 = np.flipud(mask1)
                mask2 = np.flipud(mask2)
            out_samples.append({'img': img, 'mask1': mask1, 'mask2': mask2})
        return out_samples
    return wrapper


def random_crop(output_size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):
    
    def get_params(img, output_size, mask=None):
        h,w = img.shape[0:2]
        th, tw = output_size
        range_w = 1 if w == tw else w - tw
        if h == th:
            range_h = 1
        elif mask is not None:
            rowline = np.sum(mask, 1)
            if np.sum(rowline) != 0:
                idx = np.nonzero(rowline)
                off = h - th
                range_h = min(off, idx[0][0]+1)
            else:
                range_h = 1
        else:
            range_h = h - th
        i = np.random.randint(0, range_h)
        j = np.random.randint(0, range_w)
        return i, j, th, tw
    
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            if padding is not None:
                img = np.pad(img, padding, fill, padding_mode)
                mask1 = np.pad(mask1, padding, fill, padding_mode)
                mask2 = np.pad(mask2, padding, fill, padding_mode)
    
            # pad the width if needed
            size = img.shape[0:2]
            if pad_if_needed and size[1] < output_size[1]:
                img = np.pad(img, ((0, 0), (output_size[1] - size[1], 0)),
                             fill, padding_mode)
                mask1 = np.pad(mask1, ((0, 0), (output_size[1] - size[1], 0)),
                               fill, padding_mode)
                mask2 = np.pad(mask2, ((0, 0), (output_size[1] - size[1], 0)),
                               fill, padding_mode)
    
            # pad the height if needed
            if pad_if_needed and size[0] < output_size[0]:
                img = np.pad(img, ((0, output_size[0] - size[0]),(0, 0)),
                             fill, padding_mode)
                mask1 = np.pad(mask1, ((0, output_size[0] - size[0]),(0, 0)),
                               fill, padding_mode)
                mask2 = np.pad(mask2, ((0, output_size[0] - size[0]),(0, 0)),
                               fill, padding_mode)
    
            i, j, h, w = get_params(img, output_size, mask1)

            if len(img.shape) == 3:
                img = img[i:i + h, j:j + w,:]
            else:
                img = img[i:i + h, j:j + w]
             
            mask1 = mask1[i:i + h, j:j + w]
            mask2 = mask2[i:i + h, j:j + w]
    
            out_samples.append({'img': img, 'mask1': mask1, 'mask2': mask2})
        return out_samples
    return wrapper


def normalize():
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        out_samples = []
        for sample in samples:
            img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
            img = img / 255.0
            out_samples.append({'img': img, 'mask1': mask1, 'mask2': mask2})
        return out_samples
    return wrapper


def to_tensor(nclass=12):
    def wrapper(samples):
        if type(samples) is not list:
            samples = [samples]
        # concatenate elements in samples to a single numpy array with size (N, H, W, C)
        if samples[0]['img'].ndim == 2:
            out_img = np.stack([np.expand_dims(sample['img'], 2) for sample in samples], 0)
        else:
            out_img = np.stack([sample['img'] for sample in samples], 0)
        
        onehot_mask1 = np.stack([np.expand_dims(sample['mask1'], 2) for sample in samples], 0)
        onehot_mask2 = np.stack([keras.utils.to_categorical(sample['mask2'], nclass) for sample in samples], 0)
        return (out_img, (onehot_mask1,onehot_mask2))
        
    return wrapper


if __name__ == '__main__':  
    
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    imgs = utils.listFiles('./data/images','*.png')
    msks = utils.listFiles('./data/groundtruth','*.png')
    
    my_dataset = MyPyDataset(imgs,msks,data_size=(300,200),n_class=12, batch_size=2, use_data_augmentation=True)
    
    for i in range(len(my_dataset)):
        sample = my_dataset.__getitem__(i)
        print(sample[0].shape)
        print(sample[1][0].shape)
        print(sample[1][1].shape)
        plt.figure()
        plt.subplot(131)
        plt.imshow(sample[0][0],cmap='gray')
        plt.subplot(132)
        plt.imshow(sample[1][0][0][:,:,0],cmap='gray')
        plt.subplot(133)
        plt.imshow(sample[1][1][0][:,:,4:7],cmap='gray')
        plt.show(block=True)
        plt.close()
    
    # import matplotlib.pyplot as plt
    # import numpy as np

    # t = np.arange(0.0, 2.0, 0.01)
    # s = 1 + np.sin(2*np.pi*t)
    # plt.figure()
    # plt.plot(t, s)

    # plt.title('About as simple as it gets, folks')
    # plt.show()