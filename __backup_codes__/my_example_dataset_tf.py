import tensorflow as tf
import numpy as np
import scipy.io as sio
import PIL.Image as Image
from scipy import ndimage
from utils import utils
from matplotlib import pyplot as plt
from matplotlib import cm
"""
Better performance with the tf.data
files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, num_parallel_calls=tf.data.experimental.AUTOTUNE) # This is used to remotly stored data
       .prefetch(tf.data.experimental.AUTOTUNE)
       .map(time_consuming_mapping_fuc,num_parallel_calls=tf.data.experimental.AUTOTUNE)
       .cache()
       .map(memory_consuming_mapping_fuc)
       .shuffle(100)
       .batch(2)
       .repeat()
"""

def create_2d_dataset_from_image_files(filenames, labelnames, out_shape_row_col_chn, batch_size=2, n_classes=3):
    # filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
    # labels = ["/var/data/label1.png", "/var/data/label2.png", ...]

    # read image file
    def _tf_load_image(filename, labelname):
        def _tf_read_image(image_file_name, color_model='idx'):
            def _read_image(fn, color_mode=None):
                """
                 Read image from file.
                :param image_file_name: full file path
                :param image_size_row_col: [row, col]
                :param color_mode: 'gray', 'rgb' or 'idx'
                :return: numpy image
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

            img = tf.py_function(_read_image, [image_file_name, color_model], [tf.float32])
            return tf.transpose(img, [1, 2, 0])

        out_image = _tf_read_image(filename, 'gray') / 255.0
        out_label = _tf_read_image(labelname, 'idx')
        return out_image, out_label

    # resize image
    def _tf_resize(image_decoded, label, out_size=(28, 28)):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize(image_decoded, out_size)
        return image_resized, label

    # random brightness x = x+ delta
    def _tf_random_brightness(image_decoded, label, max_delta=0.7):
        out_image = tf.image.random_brightness(image_decoded, max_delta=max_delta)
        return out_image, label

    #  random crop image
    def _tf_random_padding_crop(out_shape, data_pad_value=0, label_pad_value=0):
        def _tf_rd_crop(oct, mask):
            sz = tf.shape(oct)
            out_sz = tf.convert_to_tensor(out_shape)
            rg = sz - out_sz
            pad_value = tf.transpose(tf.stack([tf.constant([0, 0, 0]), -tf.minimum(rg, 0)]))
            oct = tf.pad(oct, paddings=pad_value, constant_values=data_pad_value)
            mask = tf.pad(mask, paddings=pad_value, constant_values=label_pad_value)
            rg1 = tf.nn.relu(rg)
            oft = tf.cast(tf.random.uniform((3,), minval=0, maxval=tf.cast(rg1, tf.float32)), tf.int32)

            oct_crop = oct[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]
            mask_crop = mask[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]

            rowline = tf.reduce_sum(mask_crop, (1, 2))
            if tf.reduce_sum(rowline) != 0:
                idx = tf.cast(tf.where(tf.not_equal(rowline, 0)), tf.int32)
                mskDeep = out_sz[0] - tf.shape(idx)[0]
                if mskDeep <= 0:
                    startp = idx[0]
                else:
                    maxv = tf.math.minimum(tf.cast(idx[0], dtype=tf.float32), tf.constant([20, ], dtype=tf.float32))
                    oft = tf.cast(tf.random.uniform((1,), minval=0, maxval=tf.cast(maxv, tf.float32)), tf.int32)
                    startp = idx[0] - oft
                endp = startp + out_sz[0]
                sz = tf.shape(oct)
                if endp[0] > sz[0]:
                    startp = sz - out_sz
                    endp = sz
                oct_crop = oct_crop[startp[0]:endp[0], :, :]
                mask_crop = mask_crop[startp[0]:endp[0], :, :]
            else:
                oct_crop = oct_crop[0:out_sz[0], :, :]
                mask_crop = mask_crop[0:out_sz[0], :, :]
            return oct_crop, mask_crop

        return _tf_rd_crop

    # random contrast
    def _tf_random_contrast(image_decoded, label_decoded, range=(0, 0.7)):
        out_image = tf.image.random_contrast(image_decoded, range[0], range[1])
        return out_image, label_decoded

    # random flip left-right
    def _tf_random_flip_left_right(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_left_right(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    # random flip up-down
    def _tf_random_flip_up_down(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_up_down(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    def _tf_random_transpose(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            transposed_image = tf.image.transpose(stacked_image)
            return transposed_image[0], transposed_image[1]
        else:
            return image_decoded, label_decoded

    # random rotate
    def _random_rotate(image):
        image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        return image

    def _tf_random_rotate(image_decoded, label):
        im_shape = image_decoded.shape
        [image, ] = tf.py_function(_random_rotate, [image_decoded], [tf.float32])
        image.set_shape(im_shape)
        return image, label

    def _tf_one_hot(nclasses):
        def _tf_one_ht(image_decoded, label):
            # image_decoded = tf.expand_dims(image_decoded)
            label_ = tf.squeeze(label, -1)
            mask_o = tf.one_hot(tf.cast(label_, dtype=tf.uint8), depth=nclasses)
            return image_decoded, mask_o

        return _tf_one_ht

    def _tf_set_shape(out_shape, n_class):
        def _set_shape(image_decoded, label):
            image_decoded = tf.reshape(image_decoded, shape=out_shape)
            label = tf.reshape(label, shape=(out_shape[0], out_shape[1], n_class))
            return image_decoded, label

        return _set_shape

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames), tf.convert_to_tensor(labelnames)))
    dataset = dataset.map(_tf_load_image)
    dataset = dataset.map(_tf_random_brightness)
    dataset = dataset.map(_tf_random_contrast)
    dataset = dataset.map(_tf_random_flip_left_right)
    dataset = dataset.map(_tf_random_flip_up_down)
    dataset = dataset.map(_tf_random_transpose)
    dataset = dataset.map(_tf_random_padding_crop(out_shape=out_shape_row_col_chn, data_pad_value=0, label_pad_value=2))
    dataset = dataset.map(_tf_one_hot(nclasses=n_classes))
    dataset = dataset.map(_tf_set_shape(out_shape=out_shape_row_col_chn, n_class=n_classes))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_2d_dataset_from_3d_mat_files(filenames, out_size=(304, 304, 304), batch_size=2, nclasses=11,
                                           contrast_range=(0.8, 1.2)):
    def _tf_read_mat(mat_file_name):
        def _read_mat(file_name):
            mat = sio.loadmat(file_name.numpy().rstrip(), verify_compressed_data_integrity=False)
            oct = mat['imgMat']
            mask = mat['maskMat']
            return oct.astype('float'), mask.astype('uint8')

        oct, mask = tf.py_function(_read_mat, [mat_file_name], [tf.float32, tf.uint8])
        return oct, mask

    #  random crop image
    def _tf_random_padding_crop(out_shape, data_pad_value=0, label_pad_value=0):
        def _tf_rd_crop(oct, mask):
            sz = tf.shape(oct)
            out_sz = tf.convert_to_tensor(out_shape)
            rg = sz - out_sz
            pad_value = tf.transpose(tf.stack([tf.constant([0, 0, 0]), -tf.minimum(rg, 0)]))
            oct = tf.pad(oct, paddings=pad_value, constant_values=data_pad_value)
            mask = tf.pad(mask, paddings=pad_value, constant_values=label_pad_value)
            rg1 = tf.nn.relu(rg)
            oft = tf.cast(tf.random.uniform((3,), minval=0, maxval=tf.cast(rg1, tf.float32)), tf.int32)

            oct_crop = oct[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]
            mask_crop = mask[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]

            rowline = tf.reduce_sum(mask_crop, (1, 2))
            if tf.reduce_sum(rowline) != 0:
                idx = tf.cast(tf.where(tf.not_equal(rowline, 0)), tf.int32)
                mskDeep = out_sz[0] - tf.shape(idx)[0]
                if mskDeep <= 0:
                    startp = idx[0]
                else:
                    maxv = tf.math.minimum(tf.cast(idx[0], dtype=tf.float32), tf.constant([20, ], dtype=tf.float32))
                    oft = tf.cast(tf.random.uniform((1,), minval=0, maxval=tf.cast(maxv, tf.float32)), tf.int32)
                    startp = idx[0] - oft
                endp = startp + out_sz[0]
                sz = tf.shape(oct)
                if endp[0] > sz[0]:
                    startp = sz - out_sz
                    endp = sz
                oct_crop = oct_crop[startp[0]:endp[0], :, :]
                mask_crop = mask_crop[startp[0]:endp[0], :, :]
            else:
                oct_crop = oct_crop[0:out_sz[0], :, :]
                mask_crop = mask_crop[0:out_sz[0], :, :]
            return oct_crop, mask_crop

        return _tf_rd_crop

    # random contrast
    def _tf_random_contrast(range=(0, 0.7)):
        def _tf_rd_contrast(oct, mask):
            oct = tf.transpose(oct, (1, 2, 0))
            oct = tf.image.random_contrast(oct, range[0], range[1])
            oct = tf.transpose(oct, (2, 0, 1))
            return oct, mask

        return _tf_rd_contrast

    # random flip left-right
    def _tf_random_flip_left_right(oct, mask):
        if tf.random.uniform(()) > 0.5:
            oct_flip = tf.image.flip_left_right(oct)
            mask_flip = tf.image.flip_left_right(mask)
            return oct_flip, mask_flip
        else:
            return oct, mask

    def _tf_random_rotate90n(oct, mask):
        id = tf.random.uniform((1,), minval=0, maxval=3, dtype=tf.int32)
        if id == 1:
            oct_t = tf.image.flip_left_right(oct)
            oct_t = tf.transpose(oct_t, (0, 2, 1))
            mask_t = tf.image.flip_left_right(mask)
            mask_t = tf.transpose(mask_t, (0, 2, 1))
        elif id == 2:
            oct_t = tf.transpose(oct, (0, 2, 1))
            oct_t = tf.image.flip_left_right(oct_t)
            oct_t = tf.transpose(oct_t, (0, 2, 1))
            mask_t = tf.transpose(mask, (0, 2, 1))
            mask_t = tf.image.flip_left_right(mask_t)
            mask_t = tf.transpose(mask_t, (0, 2, 1))
        elif id == 3:
            oct_t = tf.transpose(oct, (0, 2, 1))
            mask_t = tf.transpose(mask, (0, 2, 1))
        else:
            oct_t = oct
            mask_t = mask

        return oct_t, mask_t

    def _tf_random_extract_2d(out_size):
        def _tf_extract(oct, mask):
            if tf.random.uniform(()) > 0.5:
                idx = tf.random.uniform((1,), minval=0, maxval=out_size[1], dtype=tf.int32)
                oct_ex = tf.squeeze(oct[:, idx[0], :])
                mask_ex = tf.squeeze(mask[:, idx[0], :])
            else:
                idx = tf.random.uniform((1,), minval=0, maxval=out_size[2], dtype=tf.int32)
                oct_ex = tf.squeeze(oct[:, :, idx[0]])
                mask_ex = tf.squeeze(mask[:, :, idx[0]])

            oct_ex = tf.expand_dims(oct_ex, 2)
            oct_ex = tf.concat([oct_ex, oct_ex, oct_ex], 2)
            return oct_ex, mask_ex

        return _tf_extract

    def _tf_one_hot(nclasses):
        def _tf_one_ht(oct, mask):
            oct_o = oct / 255.0
            mask_o = tf.one_hot(mask, depth=nclasses)
            return oct_o, mask_o

        return _tf_one_ht

    def _tf_set_shape(out_shape, nclasses):
        def _set_shape(oct, mask):
            [r, c, d] = out_shape
            try:
                oct = tf.reshape(oct, shape=(r, c, 3))
                mask = tf.reshape(mask, shape=(r, c, nclasses))
            except:
                oct = tf.reshape(oct, shape=(r, d, 3))
                mask = tf.reshape(mask, shape=(r, d, nclasses))

            return oct, mask

        return _set_shape

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames),))
    dataset = dataset.map(_tf_read_mat)
    dataset = dataset.map(_tf_random_padding_crop(out_size))
    dataset = dataset.map(_tf_random_rotate90n)
    dataset = dataset.map(_tf_random_flip_left_right)
    dataset = dataset.map(_tf_random_extract_2d(out_size))
    dataset = dataset.map(_tf_one_hot(nclasses))
    dataset = dataset.map(_tf_set_shape(out_size, nclasses))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_2d_dataset_from_3d_mat_files_v2(filenames, out_size=(304, 304, 1), batch_size=2, nclasses=11,
                                           contrast_range=(0.8, 1.2)):
    def _tf_read_mat(mat_file_name):
        def _read_mat(file_name):
            mat = sio.loadmat(file_name.numpy().rstrip(), verify_compressed_data_integrity=False)
            oct = mat['imgMat']
            mask = mat['maskMat']
            return oct.astype('float'), mask.astype('uint8')

        oct, mask = tf.py_function(_read_mat, [mat_file_name], [tf.float32, tf.uint8])
        return oct, mask

    def _tf_random_extract_2d():
        def _tf_extract(oct, mask):
            sz = tf.shape(oct)
            idx = tf.random.uniform((1,), minval=0, maxval=sz[2], dtype=tf.int32)
            oct_ex = tf.squeeze(oct[:, :, idx[0]])
            mask_ex = tf.squeeze(mask[:, :, idx[0]])

            # oct_ex = tf.expand_dims(oct_ex, 2)
            # mask_ex = tf.expand_dims(mask_ex, 2)
            # oct_ex = tf.concat([oct_ex, oct_ex, oct_ex], 2)
            return oct_ex, mask_ex

        return _tf_extract

    #  random crop image
    def _tf_random_padding_crop(out_shape, data_pad_value=0, label_pad_value=0):
        def _tf_rd_crop(oct, mask):
            sz = tf.shape(oct)
            out_sz = tf.convert_to_tensor(out_shape)
            rg = sz - out_sz
            pad_value = tf.transpose(tf.stack([tf.constant([0, 0, 0]), -tf.minimum(rg, 0)]))
            oct = tf.pad(oct, paddings=pad_value, constant_values=data_pad_value)
            mask = tf.pad(mask, paddings=pad_value, constant_values=label_pad_value)
            rg1 = tf.nn.relu(rg)
            oft = tf.cast(tf.random.uniform((3,), minval=0, maxval=tf.cast(rg1, tf.float32)), tf.int32)

            oct_crop = oct[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]
            mask_crop = mask[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]

            rowline = tf.reduce_sum(mask_crop, (1, 2))
            if tf.reduce_sum(rowline) != 0:
                idx = tf.cast(tf.where(tf.not_equal(rowline, 0)), tf.int32)
                mskDeep = out_sz[0] - tf.shape(idx)[0]
                if mskDeep <= 0:
                    startp = idx[0]
                else:
                    maxv = tf.math.minimum(tf.cast(idx[0], dtype=tf.float32), tf.constant([20, ], dtype=tf.float32))
                    oft = tf.cast(tf.random.uniform((1,), minval=0, maxval=tf.cast(maxv, tf.float32)), tf.int32)
                    startp = idx[0] - oft
                endp = startp + out_sz[0]
                sz = tf.shape(oct)
                if endp[0] > sz[0]:
                    startp = sz - out_sz
                    endp = sz
                oct_crop = oct_crop[startp[0]:endp[0], :, :]
                mask_crop = mask_crop[startp[0]:endp[0], :, :]
            else:
                oct_crop = oct_crop[0:out_sz[0], :, :]
                mask_crop = mask_crop[0:out_sz[0], :, :]
            return oct_crop, mask_crop

        return _tf_rd_crop

    # random contrast
    def _tf_random_contrast(range=(0, 0.7)):
        def _tf_rd_contrast(oct, mask):
            oct = tf.transpose(oct, (1, 2, 0))
            oct = tf.image.random_contrast(oct, range[0], range[1])
            oct = tf.transpose(oct, (2, 0, 1))
            return oct, mask

        return _tf_rd_contrast

    # random flip left-right
    def _tf_random_flip_left_right(oct, mask):
        if tf.random.uniform(()) > 0.5:
            oct_flip = tf.image.flip_left_right(oct)
            mask_flip = tf.image.flip_left_right(mask)
            return oct_flip, mask_flip
        else:
            return oct, mask

    def _tf_one_hot(nclasses):
        def _tf_one_ht(oct, mask):
            oct_ex = tf.expand_dims(oct, 2)
            oct_ex = tf.concat([oct_ex, oct_ex, oct_ex], 2)
            oct_o = oct_ex / 255.0
            mask_o = tf.one_hot(mask, depth=nclasses)
            return oct_o, mask_o

        return _tf_one_ht

    def _tf_set_shape(out_shape, nclasses):
        def _set_shape(oct, mask):
            [r, c, d] = out_shape
            oct = tf.reshape(oct, shape=(r, c, 3))
            mask = tf.reshape(mask, shape=(r, c, nclasses))
            return oct, mask

        return _set_shape

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames),))
    dataset = dataset.map(_tf_read_mat)
    dataset = dataset.map(_tf_random_extract_2d)
    dataset = dataset.map(_tf_random_padding_crop(out_size))
    dataset = dataset.map(_tf_random_flip_left_right)

    dataset = dataset.map(_tf_one_hot(nclasses))
    dataset = dataset.map(_tf_set_shape(out_size, nclasses))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def create_3d_dataset_from_3d_mat_files(filenames, out_size, batch_size=2, nclasses=11,
                                        contrast_range=(0, 0.7)):
    def _tf_read_mat(mat_file_name):
        def _read_mat(file_name):
            mat = sio.loadmat(file_name.numpy().rstrip(), verify_compressed_data_integrity=False)
            oct = mat['imgMat']
            mask = mat['maskMat']
            return oct.astype('float'), mask.astype('uint8')

        oct, mask = tf.py_function(_read_mat, [mat_file_name], [tf.float32, tf.uint8])
        return oct, mask

    #  random crop image
    def _tf_random_padding_crop(out_shape, data_pad_value=0, label_pad_value=0):
        def _tf_rd_crop(oct, mask):
            sz = tf.shape(oct)
            out_sz = tf.convert_to_tensor(out_shape)
            rg = sz - out_sz
            pad_value = tf.transpose(tf.stack([tf.constant([0, 0, 0]), -tf.minimum(rg, 0)]))
            oct = tf.pad(oct, paddings=pad_value, constant_values=data_pad_value)
            mask = tf.pad(mask, paddings=pad_value, constant_values=label_pad_value)
            rg1 = tf.nn.relu(rg)
            oft = tf.cast(tf.random.uniform((3,), minval=0, maxval=tf.cast(rg1, tf.float32)), tf.int32)

            oct_crop = oct[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]
            mask_crop = mask[:, oft[1]:oft[1] + out_sz[1], oft[2]:oft[2] + out_sz[2]]

            rowline = tf.reduce_sum(mask_crop, (1, 2))
            idx = tf.cast(tf.where(tf.not_equal(rowline, 0)), tf.int32)
            mskDeep = out_sz[0] - tf.shape(idx)[0]
            if mskDeep <= 0:
                startp = idx[0]
            else:
                maxv = tf.math.minimum(tf.cast(idx[0], dtype=tf.float32), tf.constant([20, ], dtype=tf.float32))
                oft = tf.cast(tf.random.uniform((1,), minval=0, maxval=tf.cast(maxv, tf.float32)), tf.int32)
                startp = idx[0] - oft
            endp = startp + out_sz[0]
            if endp[0] > sz[0]:
                startp = sz - out_sz
                endp = sz
            oct_crop = oct_crop[startp[0]:endp[0], :, :]
            mask_crop = mask_crop[startp[0]:endp[0], :, :]
            return oct_crop, mask_crop

        return _tf_rd_crop

    # random contrast
    def _tf_random_contrast(range=(0, 0.7)):
        def _tf_rd_contrast(oct, mask):
            oct = tf.transpose(oct, (1, 2, 0))
            oct = tf.image.random_contrast(oct, range[0], range[1])
            oct = tf.transpose(oct, (2, 0, 1))
            return oct, mask

        return _tf_rd_contrast

    # random flip left-right
    def _tf_random_flip_left_right(oct, mask):
        if tf.random.uniform(()) > 0.5:
            oct_flip = tf.image.flip_left_right(oct)
            mask_flip = tf.image.flip_left_right(mask)
            return oct_flip, mask_flip
        else:
            return oct, mask

    def _tf_random_rotate90n(oct, mask):
        id = tf.random.uniform((1,), minval=0, maxval=3, dtype=tf.int32)
        if id == 1:
            oct_t = tf.image.flip_left_right(oct)
            oct_t = tf.transpose(oct_t, (0, 2, 1))
            mask_t = tf.image.flip_left_right(mask)
            mask_t = tf.transpose(mask_t, (0, 2, 1))
        elif id == 2:
            oct_t = tf.transpose(oct, (0, 2, 1))
            oct_t = tf.image.flip_left_right(oct_t)
            oct_t = tf.transpose(oct_t, (0, 2, 1))
            mask_t = tf.transpose(mask, (0, 2, 1))
            mask_t = tf.image.flip_left_right(mask_t)
            mask_t = tf.transpose(mask_t, (0, 2, 1))
        elif id == 3:
            oct_t = tf.transpose(oct, (0, 2, 1))
            mask_t = tf.transpose(mask, (0, 2, 1))
        else:
            oct_t = oct
            mask_t = mask

        return oct_t, mask_t

    def _tf_one_hot(nclasses):
        def _tf_one_ht(oct, mask):
            oct_o = tf.expand_dims(oct, 3) / 255.0
            mask_o = tf.one_hot(mask, depth=nclasses)
            return oct_o, mask_o

        return _tf_one_ht

    def _tf_set_shape(out_shape, nclasses):
        def _set_shape(oct, mask):
            oct = tf.reshape(oct, shape=(*out_shape, 1))
            mask = tf.reshape(mask, shape=(*out_shape, nclasses))
            return oct, mask

        return _set_shape

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames),))
    dataset = dataset.map(_tf_read_mat)
    dataset = dataset.map(_tf_random_rotate90n)
    dataset = dataset.map(_tf_random_flip_left_right)
    dataset = dataset.map(_tf_random_padding_crop(out_size))
    dataset = dataset.map(_tf_one_hot(nclasses))
    dataset = dataset.map(_tf_set_shape(out_size, nclasses))

    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    return dataset


def example_create_dataset_from_npy(npy_file_path):
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load(npy_file_path) as data:
        features = data["features"]
        labels = data["labels"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset


def example_create_dataset_from_tfrecord(tfrecord_files):
    """"
    Creates a dataset that reads all of the examples from two files.
    training_dataset = example_create_dataset_from_tfrecord(["/training1.tfrecord", "/training1.tfrecord"])
    validation_dataset = example_create_dataset_from_tfrecord(["/validation1.tfrecord", "/validation1.tfrecord"])
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(...)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)
    return dataset


def example_create_dataset_from_txt(txt_files):
    """"
    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    #
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    """
    dataset = tf.data.Dataset.from_tensor_slices(txt_files)
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
                .skip(1)
                .filter(lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), "#"))))
    return dataset


if __name__ == '__main__':

    img_list = utils.listFiles('./data/images','*.png')
    gts_list = utils.listFiles('./data/groundtruth','*.png') 
    
    dataset = create_2d_dataset_from_image_files(img_list, gts_list, out_shape_row_col_chn=(480, 304, 1), batch_size=1, n_classes=12)
    
    for ele in dataset:
        [img,gt] = ele
        img = np.squeeze(img.numpy())
        gt = np.squeeze(gt.numpy())
        img = np.uint8(img*255)
        gt = np.uint8(gt[:,:,0:3]*255)
        # im = Image.fromarray(np.uint8(cm.gist_earth(img)*255))
        # im.save('1.png')
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(gt)
        plt.savefig('1.png')