import numpy as np
import scipy.io as sio
import PIL.Image as Image
import glob
import os
import cv2
import natsort

def read_file_list(list_txt_file):
    fp = open(list_txt_file, 'r')
    files = fp.readlines()
    files = [item.rstrip() for item in files]
    return files


def listFiles(folder, file_filter="**/*", recursive=True):
    return natsort.natsorted(list(glob.iglob(os.path.join(folder, file_filter), recursive=recursive)))


def split_list(file_list, split=(0.8, 0.2, 0), shuffle=False):
    if shuffle:
        np.random.shuffle(file_list)
    image_num = len(file_list)
    split_idx = [int(np.ceil(element * image_num)) for element in split]
    s = sum(split_idx[:2])
    train_list = file_list[:split_idx[0]]
    test_list = file_list[split_idx[0]:s]
    valid_list = file_list[s:]
    return train_list, test_list, valid_list


def k_fold_split(file_list, fold=5):
    kFold_list_file = []
    kFold_list_idx = []
    data_set_length = len(file_list)
    index = np.tile(np.arange(0, fold), data_set_length //
                    fold + 1)[:data_set_length]
    for i in range(fold):
        flg_valid = np.where(index == i)[0]
        flg_train = np.where(index != i)[0]
        train_lst = [file_list[j] for j in flg_valid]
        Valid_lst = [file_list[j] for j in flg_train]
        kFold_list_idx.append([flg_train, flg_valid])
        kFold_list_file.append([train_lst, Valid_lst])
    return kFold_list_file, kFold_list_idx


def shuffle_lists(*file_lists):
    """
    Example:
        list1=['a', 'b', 'c']
        list2=[1, 2, 3]
        list1_s,list2_s=shuffle_data_files(list1,list2)
        list1_s = ['a', 'c', 'b']
        list2_s = [1, 3, 2]
    :param file_lists: any numbers of list
    :return: shuffled lists
    """
    if len(file_lists) == 1:
        list_files = list(*file_lists)
        np.random.shuffle(list_files)
        return list_files
    else:
        list_files = list(zip(*file_lists))
        np.random.shuffle(list_files)
        return zip(*list_files)


def read_image(fn, color_mode=None):
    """
     Read image from file.
    :param image_file_name: full file path
    :param image_size_row_col: [row, col]
    :param color_mode: 'gray', 'rgb' or 'idx'
    :return: numpy image
    """
    img = Image.open(fn.rstrip())
    if color_mode is not None:
        if color_mode.lower() == 'gray':
            img = img.convert('L')
        else:
            if color_mode.lower() == 'rgb':
                img = img.convert('RGB')
            else:
                if color_mode.lower() == 'idx':
                    img = img.convert('P')
    return np.array(img)


def read_mat_list_to_npy(file_list, shuffle=True):
    npy_data = []
    if shuffle:
        np.random.shuffle(file_list)
    for f in file_list:
        mat = sio.loadmat(f)
        img, mask = np.array(mat['imgMat'], dtype='float32'), np.array(
            mat['maskMat'], dtype='int64')

        npy_data.extend((img[:, :, i], mask[:, :, i])
                        for i in range(img.shape[2]))
        print(f'reading {f}')
    return npy_data


def read_img_list_to_npy(file_list, color_mode, shuffle=False):
    def _read_image(image_file_name, mode=None):
        """
         Read image from file.
        :param image_file_name: full file path
        :param mode: 'gray', 'rgb' or 'idx'
        :return: numpy image
        """
        img = Image.open(image_file_name.rstrip())
        if mode is not None:
            if mode.lower() == 'gray':
                img = img.convert('L')
            else:
                if mode.lower() == 'rgb':
                    img = img.convert('RGB')
                else:
                    if mode.lower() == 'idx':
                        img = img.convert('P')
        return np.asarray(img, dtype='float32')
    npy_data = []
    l = len(file_list)
    if shuffle:
        np.random.shuffle(file_list)
    for i, f in enumerate(file_list):
        im = _read_image(f, color_mode)
        npy_data.append(im)
        print('reading {:.2f}%'.format(i/l*100))
    return npy_data


def split_dataset_npy(npy_file, split_ratio=(0.8, 0.2, 0), split_idx=None, shuffle=True):
    """
    :param npy_file: 'data.npy'
    :param split_ratio: (train,test,valid) = (0.8, 0.2, 0)
    :param split_idx: example: split to two set train and test [[1,3,5,7,9],[0,2,4,6,8]]
    :param shuffle: True or False
    :return: train_set, test_set, valid_set, ...
    """
    data = np.load(npy_file, allow_pickle=True)
    split_data = []
    if not split_idx:
        if shuffle:
            np.random.shuffle(data)
        data_num = len(data)
        split_rg = [int(np.ceil(element * data_num))
                    for element in split_ratio]
        s = sum(split_rg[:2])
        train = data[:split_rg[0]]
        test = data[split_rg[0]:s]
        valid = data[s:]
    else:
        for idx in split_idx:
            if shuffle:
                tmp = [data[i] for i in idx]
                np.random.shuffle(tmp)
                split_data.append(tmp)
            else:
                split_data.append([data[i] for i in idx])
        train, test, *valid = split_data
    return train, test, valid

def pad_pow2_size(img, pow2=4):
    scale = np.power(2, pow2)
    r = np.ceil(img.shape[0] / scale) * scale
    c = np.ceil(img.shape[1] / scale) * scale
    pad_width = [[0, (r - img.shape[0]).astype(int)],
                 [0, (c - img.shape[1]).astype(int)]]
    return np.pad(img, pad_width)


def restore_padded_pow2_size(img, raw_size):
    return img[:raw_size[0], :raw_size[1]]

def resize_pow2_size(img, pow2=5,min_size=None):
    size_pow2 = np.power(2, pow2)
    nh, nw = img.shape[0:2]
    offset_r = nh % size_pow2
    offset_c = nw % size_pow2
    if offset_r > 0:
        offset_r = size_pow2 - offset_r
    if offset_c > 0:
        offset_c = size_pow2 - offset_c
    if min_size is not None:
        new_h = max(nh + offset_r,min_size[0])
        new_w = max(nw + offset_c,min_size[1])
    else:
        new_h = max(nh + offset_r, size_pow2)
        new_w = max(nw + offset_c, size_pow2)
    
    img_r = cv2.resize(img, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
    # if len(img.shape) == 2:
    #     img_r = np.expand_dims(np.expand_dims(img_r, 2), 0)
    # else:
    #     img_r = np.expand_dims(img_r, 0)
    return img_r

def restore_resized_pow2size(img, raw_size):
    return cv2.resize(img, dsize=(raw_size[1],raw_size[0]), interpolation=cv2.INTER_NEAREST)   

def build_montages(image_list, image_shape, montage_shape):
    """
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------

    example usage:

    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = build_montages(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)

    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception(
            'image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception(
            'montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    montage_image = np.zeros(shape=(
        image_shape[1] * montage_shape[1], image_shape[0] * montage_shape[0], 3), dtype=np.uint8)

    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception(
                f'input of type {type(img)} is not a valid numpy array')

        start_new_img = False
        img = cv2.resize(img, image_shape)

        # check if the channel of image is 1, if so, convert to 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        montage_image[cursor_pos[1]: cursor_pos[1] + image_shape[1],
                      cursor_pos[0]: cursor_pos[0] + image_shape[0],:] = img

        cursor_pos[0] += image_shape[0]
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                montage_image = np.zeros(shape=(
                    image_shape[1] * montage_shape[1], image_shape[0] * montage_shape[0], 3), dtype=np.uint8)

                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)
    return image_montages


def apply_colormap(im_gray):
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.asarray([
        [0, 0, 0], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [232, 48, 0], [0, 152, 255], [94, 255, 3], [133, 2, 255],
    [4, 247, 87], [255, 181, 0], [6, 227, 196], [218, 0, 4], [231, 4, 196], [0, 0, 166], [255, 155, 3], [5, 104, 236], [188, 35, 255], [255, 1, 105],
    [0, 198, 200], [54, 61, 255], [255, 111, 1], [0, 205, 187], [255, 116, 254], [164, 232, 4], [226, 188, 0], [198, 211, 0], [12, 234, 145], [226, 0, 39],
    [0, 248, 179], [2, 155, 219], [0, 69, 210], [205, 0, 197], [0, 216, 145], [255, 244, 109], [2, 211, 70], [0, 182, 197], [246, 226, 227], [42, 127, 255],
    [218, 76, 255], [136, 255, 236], [255, 246, 159], [10, 166, 216], [239, 175, 255], [186, 9, 0], [236, 82, 0], [255, 160, 242], [181, 6, 211], [255, 90, 228],
    [121, 0, 215], [59, 93, 255], [234, 0, 114], [132, 237, 247], [252, 228, 200], [227, 0, 145], [255, 26, 89], [190, 221, 255], [0, 96, 205], [0, 128, 207],
    [0, 49, 9], [221, 215, 243], [127, 222, 254], [231, 4, 82], [0, 194, 160], [255, 69, 38], [46, 181, 0], [229, 253, 164], [0, 180, 51], [255, 59, 193],
    [50, 46, 223], [219, 203, 246], [255, 179, 225], [255, 132, 230], [211, 191, 255], [218, 113, 255], [7, 92, 0], [255, 228, 125], [1, 64, 11], [223, 251, 113],
    [146, 0, 3], [79, 198, 1], [0, 46, 23], [244, 215, 73], [65, 6, 1], [0, 145, 190], [1, 134, 21], [255, 224, 158], [254, 214, 189], [210, 0, 150],
    [128, 255, 205], [27, 225, 119], [218, 0, 124], [140, 208, 255], [200, 208, 246], [234, 28, 169], [156, 204, 4], [215, 144, 255], [204, 233, 58],
    [226, 122, 5], [190, 0, 40], [201, 226, 230], [209, 172, 254], [48, 0, 24], [0, 166, 170], [214, 142, 1], [90, 0, 7], [83, 110, 255], [0, 44, 39],
    [194, 48, 0], [0, 154, 46], [166, 0, 25], [0, 25, 90], [181, 180, 0], [178, 194, 254], [99, 255, 172], [94, 167, 255], [255, 47, 128], [2, 81, 23],
    [0, 181, 127], [207, 246, 180], [27, 68, 0], [25, 24, 27], [0, 171, 77], [185, 3, 170], [2, 34, 123], [173, 255, 96], [255, 59, 83], [140, 242, 212],
    [255, 104, 50], [57, 20, 6], [194, 255, 153], [209, 97, 0], [157, 235, 221], [49, 221, 174], [175, 216, 236], [96, 142, 255], [14, 114, 197],
    [11, 33, 44], [197, 151, 0], [30, 110, 0], [247, 201, 191], [204, 7, 68], [254, 178, 198], [255, 74, 70], [182, 228, 222], [199, 210, 231],
    [0, 46, 56], [202, 232, 206], [254, 201, 109], [55, 33, 1], [255, 181, 80], [254, 165, 202], [230, 229, 167], [16, 24, 53], [3, 38, 65],
    [173, 170, 255], [250, 208, 159], [0, 72, 156], [231, 219, 188], [198, 0, 90], [3, 173, 137], [0, 75, 40], [143, 176, 255], [12, 189, 102],
    [255, 144, 201], [255, 110, 194], [110, 255, 146], [0, 137, 163], [50, 0, 51], [212, 233, 185], [180, 162, 0], [227, 131, 230], [220, 186, 227],
    [49, 86, 220], [174, 129, 255], [102, 225, 211], [255, 145, 63], [0, 111, 166], [0, 49, 119], [0, 164, 95], [121, 219, 33], [59, 151, 0], [190, 71, 0],
    [2, 60, 50], [32, 22, 37], [127, 158, 255], [55, 45, 0], [139, 180, 0], [6, 126, 175], [1, 44, 88], [92, 1, 26], [228, 81, 209], [255, 186, 173],
    [163, 218, 228], [178, 234, 206], [122, 123, 255], [227, 170, 224], [86, 26, 2], [32, 34, 26], [220, 222, 92], [122, 0, 29], [1, 51, 73], [156, 255, 147],
    [113, 178, 245], [45, 32, 17], [1, 72, 51], [255, 192, 127], [0, 108, 49], [32, 55, 14], [78, 0, 37], [0, 137, 65], [120, 200, 235], [143, 93, 248],
    [239, 191, 196], [3, 145, 154], [37, 14, 53], [179, 0, 139], [196, 34, 33], [51, 125, 0], [0, 155, 117], [31, 42, 26], [0, 118, 153], [54, 22, 24],
    [185, 0, 118], [255, 131, 71], [133, 2, 170], [255, 93, 167], [0, 68, 125], [0, 144, 135], [210, 196, 219], [183, 5, 70], [30, 35, 36], [0, 96, 57],
    [220, 206, 201], [0, 69, 71], [75, 44, 0], [255, 183, 137], [255, 168, 97], [0, 77, 67], [186, 98, 0], [152, 0, 52], [0, 144, 94], [41, 32, 29],
    [30, 32, 43], [58, 63, 0], [50, 88, 0], [49, 29, 25], [255, 79, 120], [27, 42, 37], [0, 92, 139], [210, 22, 86], [119, 38, 0], [255, 142, 177],
    [55, 143, 219]
    ])
    return np.array(colormap[im_gray], dtype='uint8')


def find_best_row_col(n):
    t = np.int(np.ceil(np.sqrt(n)))
    rg = np.arange(1, t + 1)
    cols, rows = 1, t
    for i in rg[::-1]:
        if n % i == 0:
            rows = i
            cols = np.int(n / rows)
            break
    cols, rows = [rows, cols] if cols < rows else [cols, rows]
    rows, cols = [np.int(np.ceil(n / t)),
                  t] if rows == 1 or cols / rows > 3 else [rows, cols]
    return rows, cols


def pad_power2_size(img, downsample_level=4):
    scale = np.power(2, downsample_level)
    r = np.ceil(img.shape[0] / scale) * scale
    c = np.ceil(img.shape[1] / scale) * scale
    pad_width = [[0, (r - img.shape[0]).astype(int)],
                 [0, (c - img.shape[1]).astype(int)]]
    return np.pad(img, pad_width)


def restore_size(img, raw_size):
    return img[:raw_size[0], :raw_size[1]]


def split3d(mat, split_type, sections=(1, 1, 1), block_sz=None, overlap_sz=(0, 0, 0), make_up_resdual_section=True):
    '''
        :param make_up_resdual_section:
        :param overlap_sz:
        :param block_sz:
        :param mat: numpy ndarray
        :param sections:
        :param split_type: 'section' or 'fix_size'
        :return:
    '''
    in_shape = np.array(np.array(mat).shape)
    indices = []
    blocks = []

    def _get_idcs(sect, bsz, opsz, insp, mkup):
        for i in range(sect[0]):
            for j in range(sect[1]):
                for k in range(sect[2]):
                    std_r, end_r = (i * bsz[0] - i * opsz[0], (i + 1) * bsz[0] - i * opsz[0])
                    if end_r > insp[0]:
                        if mkup:
                            std_r, end_r = insp[0] - bsz[0], insp[0]
                        else:
                            end_r = insp[0]

                    std_c, end_c = (j * bsz[1] - j * opsz[1], (j + 1) * bsz[1] - j * opsz[1])
                    if end_c > insp[1]:
                        if mkup:
                            std_c, end_c = insp[1] - bsz[1], insp[1]
                        else:
                            end_c = insp[1]

                    std_d, end_d = (k * bsz[2] - k * opsz[2], (k + 1) * bsz[2] - k * opsz[2])
                    if end_d > insp[2]:
                        if mkup:
                            std_d, end_d = insp[2] - bsz[2], insp[2]
                        else:
                            end_d = insp[2]

                    idc = np.array([std_r, end_r, std_c, end_c, std_d, end_d], dtype='int32')
                    indices.append(idc)
                    blocks.append(mat[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]])
        return blocks, indices

    if split_type == 'section':
        sect = in_shape / np.array(sections)
        block_sz = np.ceil(sect)
        blocks, indices = _get_idcs(sections, block_sz, (0, 0, 0), in_shape, make_up_resdual_section)

    elif split_type == 'fix_size':
        block_sz = np.array(block_sz)
        block_sz = np.minimum(in_shape, block_sz)

        if np.sum(np.abs(np.array(overlap_sz))) == 0:
            sections = np.array(np.ceil(in_shape / block_sz), dtype='int32')
        else:
            overlap_sz = np.array(overlap_sz)
            ovp = block_sz - overlap_sz
            ovp[ovp == 0] = 1
            sections = np.array(np.ceil((in_shape - block_sz) / ovp) + 1,
                                dtype='int32')
        blocks, indices = _get_idcs(sections, block_sz, overlap_sz, in_shape, make_up_resdual_section)
    indices.insert(0, in_shape)
    return blocks, indices


def concatenate3d(blocks, indices):
    total_size = indices[0]
    indice = indices[1:]
    mat = np.zeros(total_size)
    msk = np.zeros(total_size)
    for blk, idc in zip(blocks, indice):
        mat[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] = mat[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] + blk
        msk[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] = msk[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] + 1
    return mat / msk


def split4d(mat, split_type, sections=(1, 1, 1, 1), block_sz=None, overlap_sz=(0, 0, 0, 0),
            make_up_resdual_section=True):
    '''
        :param make_up_resdual_section:
        :param overlap_sz:
        :param block_sz:
        :param mat: numpy ndarray
        :param sections:
        :param split_type: 'section' or 'fix_size'
        :return:
    '''
    in_shape = np.array(np.array(mat).shape)
    indices = []
    blocks = []

    def _get_idcs(sect, bsz, opsz, insp, mkup):
        for i in range(sect[0]):
            for j in range(sect[1]):
                for k in range(sect[2]):
                    for t in range(sect[3]):
                        std_r, end_r = (i * bsz[0] - i * opsz[0], (i + 1) * bsz[0] - i * opsz[0])
                        if end_r > insp[0]:
                            if mkup:
                                std_r, end_r = insp[0] - bsz[0], insp[0]
                            else:
                                end_r = insp[0]

                        std_c, end_c = (j * bsz[1] - j * opsz[1], (j + 1) * bsz[1] - j * opsz[1])
                        if end_c > insp[1]:
                            if mkup:
                                std_c, end_c = insp[1] - bsz[1], insp[1]
                            else:
                                end_c = insp[1]

                        std_d, end_d = (k * bsz[2] - k * opsz[2], (k + 1) * bsz[2] - k * opsz[2])
                        if end_d > insp[2]:
                            if mkup:
                                std_d, end_d = insp[2] - bsz[2], insp[2]
                            else:
                                end_d = insp[2]

                        std_t, end_t = (t * bsz[3] - t * opsz[3], (t + 1) * bsz[3] - t * opsz[3])
                        if end_t > insp[3]:
                            if mkup:
                                std_t, end_t = insp[3] - bsz[3], insp[3]
                            else:
                                end_d = insp[3]

                        idc = np.array([std_r, end_r, std_c, end_c, std_d, end_d, std_t, end_t], dtype='int32')
                        indices.append(idc)
                        blocks.append(mat[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5], idc[6]:idc[7]])
        return blocks, indices

    if split_type == 'section':
        sect = in_shape / np.array(sections)
        block_sz = np.ceil(sect)
        blocks, indices = _get_idcs(sections, block_sz, (0, 0, 0, 0), in_shape, make_up_resdual_section)

    elif split_type == 'fix_size':
        block_sz = np.array(block_sz)
        block_sz = np.minimum(in_shape, block_sz)

        if np.sum(np.abs(np.array(overlap_sz))) == 0:
            sections = np.array(np.ceil(in_shape / block_sz), dtype='int32')
        else:
            overlap_sz = np.array(overlap_sz)
            ovp = block_sz-overlap_sz
            ovp[ovp == 0] = 1
            sections = np.array(np.ceil((in_shape - block_sz) / ovp) + 1,
                                dtype='int32')
        blocks, indices = _get_idcs(sections, block_sz, overlap_sz, in_shape, make_up_resdual_section)
    indices.insert(0, in_shape)
    return blocks, indices


def concatenate4d(blocks, indices, total_size=None, change_channel=0):
    if change_channel:
        for i in range(len(indices)):
            indices[i][-1] = change_channel
    if not total_size:
        total_size = indices[0]
    indic = indices[1:]
    mat = np.zeros(total_size)
    msk = np.zeros(total_size)
    for blk, idc in zip(blocks, indic):
        mat[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5], idc[6]:idc[7]] = mat[idc[0]:idc[1], idc[2]:idc[3],
                                                                          idc[4]:idc[5], idc[6]:idc[7]] + blk
        msk[idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5], idc[6]:idc[7]] = msk[idc[0]:idc[1], idc[2]:idc[3],
                                                                          idc[4]:idc[5], idc[6]:idc[7]] + 1
    return mat / msk


if __name__ == "__main__":
    path = 'F:\Data4LayerSegmentation\_Dataset_v2_'
    print(listFiles(path, '**/*.mat'))
