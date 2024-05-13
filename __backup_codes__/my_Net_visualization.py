from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys


class MyNetVisualization(object):

    @staticmethod
    def cvt_imgarray2image(img_array, row=None, col=None, margin=2, data_format='channel_last', preprocess=False):
        if data_format != 'channel_last':
            img_array = img_array.transpose((3, 1, 2, 0))
        sz = img_array.shape

        if row is None or col is None:
            row, col = MyNetVisualization.find_best_row_col(sz[0])

        out_img = np.zeros((row * sz[1] + (row + 1) * margin, col * sz[2] + (col + 1) * margin, sz[3])).astype('uint8')
        k = 0
        for i in range(row):
            for j in range(col):
                h_start = i * sz[1] + (i + 1) * margin
                h_end = h_start + sz[1]
                v_start = j * sz[2] + (j + 1) * margin
                v_end = v_start + sz[2]
                if k >= sz[0]:
                    break
                im = img_array[k, :, :, :]
                if preprocess:
                    im -= im.mean()
                    im /= (im.std() + keras.backend.epsilon())
                    im *= 64
                    im += 128
                    im = np.clip(im, 0, 255).astype('uint8')
                out_img[h_start:h_end, v_start:v_end, :] = im
                img_array[k, :, :, :] = im
                k += 1
        return np.squeeze(out_img), img_array

    @staticmethod
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
        rows, cols = [np.int(np.ceil(n / t)), t] if rows == 1 or cols / rows > 3 else [rows, cols]
        return rows, cols

    @staticmethod
    def normalize(x):
        """utility function to normalize a tensor.

        # Arguments
            x: An input tensor.

        # Returns
            The normalized input tensor.
        """
        return x / (keras.backend.sqrt(keras.backend.mean(keras.backend.square(x))) + keras.backend.epsilon())

    @staticmethod
    def deprocess_image(x, apply_gaussian=False, ksize=(3, 3), sigma_x=0):
        """utility function to convert a float array into a valid uint8 image.

        # Arguments
            x: A numpy-array representing the generated image.

        # Returns
            A processed numpy-array, which could be used in e.g. imshow.
        """
        # normalize tensor: center on 0., ensure std is 0.25
        x -= x.mean()
        x /= (x.std() + keras.backend.epsilon())
        x *= 0.25

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if keras.backend.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        if apply_gaussian:
            x = cv2.GaussianBlur(x, ksize, sigma_x)
            x = np.expand_dims(x, 3) if len(x.shape) < 3 else x
        return x

    @staticmethod
    def process_image(x, former):
        """utility function to convert a valid uint8 image back into a float array.
           Reverses `deprocess_image`.

        # Arguments
            x: A numpy-array, which could be used in e.g. imshow.
            former: The former numpy-array.
                    Need to determine the former mean and variance.

        # Returns
            A processed numpy-array representing the generated image.
        """
        if keras.backend.image_data_format() == 'channels_first':
            x = x.transpose((2, 0, 1))
        return (x / 255 - 0.5) * 4 * former.std() + former.mean()

    @staticmethod
    def load_model(model_path, custom_objects=None, show_summary=True):
        m = keras.models.load_model(filepath=model_path, custom_objects=custom_objects, compile=True)
        if show_summary:
            m.summary()
        return m

    @staticmethod
    def show_featuremaps(model, layer_name, img_path, preproces_fun, is_show=True, show_index_range=None):
        """

        :param model:
        :param layer_name:
        :param img_path:
        :param preproces_fun:
        :param is_show:
        :param show_index_range:
        :return:
        """
        model_input = preproces_fun(img_path)

        # this is the placeholder for the input images
        input_img = model.input
        try:
            # this is the placeholder for the conv output
            out_conv = model.get_layer(layer_name).output
        except:
            raise Exception('Not layer named {}!'.format(layer_name))
        # get the intermediate layer model
        intermediate_layer_model = keras.models.Model(inputs=input_img, outputs=out_conv)
        # get the output of intermediate layer model
        out_feature_maps = intermediate_layer_model.predict(model_input)
        out_feature_maps = np.transpose(out_feature_maps, (3, 1, 2, 0))
        if show_index_range is not None:
            out_feature_maps = out_feature_maps[show_index_range[0]:show_index_range[1], :, :, :]
        image, out_feature_maps = MyNetVisualization.cvt_imgarray2image(out_feature_maps, margin=2, preprocess=True)
        if is_show:
            plt.figure(num='Convolution kernel of \'' + layer_name + '\'')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
            plt.axis('off')
            plt.imshow(image)
            plt.show()
        return image, out_feature_maps

    @staticmethod
    def show_conv_kernel(model, layer_name, input_layer_names=None, model_input_mean=128., model_input_shift=20.,
                         is_show=True, show_index_range=None, iteration=200, verbose=1, apply_gaussian=True,
                         out_all=False):
        """

        :param model: keras model
        :param layer_name:
        :param input_layer_names:
        :param model_input_mean:
        :param model_input_shift:
        :param is_show:
        :param show_index_range:
        :param iteration:
        :param verbose:
        :param apply_gaussian:
        :param out_all:
        :return:
        """
        # this is the placeholder for the input images
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        if input_layer_names is None:
            input_img = model.input
            if isinstance(input_img, list):
                input_layer_names = [ipt.op.name for ipt in input_img]
            else:
                input_layer_names = [input_img.op.name]
                input_img = [input_img]
        else:
            if type(input_layer_names) == str:
                input_layer_names = (input_layer_names,)
            try:
                input_img = [layer_dict[name].input for name in input_layer_names]
            except:
                raise Exception('One or more layer names are not find!'.format(input_layer_names))

        input_img_shape = [ipt.keras_shape[1:] for ipt in input_img]

        try:
            layer_output = layer_dict[layer_name].output
        except:
            raise Exception('Not layer named {}!'.format(layer_name))

        kept_filters = []
        if show_index_range is None:
            shp = layer_output.shape
            show_index_range = (0, shp[-1].value)
        show_number = show_index_range[1] - show_index_range[0]
        for i in range(show_index_range[0], show_index_range[1]):
            loss = keras.backend.mean(layer_output[:, :, :, i])

            # compute the gradient of the input picture with this loss
            grads = keras.backend.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = MyNetVisualization.normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = keras.backend.function(input_img, [loss, grads])

            # step size for gradient ascent
            step = 1.
            # run gradient ascent for iteration steps
            input_img_data = [np.random.random((1,) + shape) * model_input_shift + model_input_mean
                              for shape in input_img_shape]
            # input_img_data = [np.random.random((1,) + input_img_shape[0]) * model_input_shift + model_input_mean]
            loss_value = None
            for j in range(iteration):
                loss_value, grads_value = iterate(input_img_data)
                input_img_data = [ig + grads_value * step for ig in input_img_data]

                if verbose == 1:
                    print('\rKernel %03d: %03d, gradient ascent iteration: %03d: %03d' % (
                        show_number, i + 1, iteration, j + 1))
                elif verbose == 2:
                    sys.stdout.write(
                        '\rKernel %03d: %03d, gradient ascent iteration: %03d: %03d' % (
                            show_number, i + 1, iteration, j + 1))
                    sys.stdout.flush()

            # decode the resulting input image

            input_img_data = [MyNetVisualization.deprocess_image(img_[0], apply_gaussian) for img_ in input_img_data]

            kept_filters.append((input_img_data, loss_value))

            # sort filter result
            kept_filters.sort(key=lambda x: x[1], reverse=True)

        out_filters = []
        out_imgs = []
        if out_all:
            for i in range(len(input_img)):
                filters = np.array([f[0][i] for f in kept_filters])
                img_e, filters = MyNetVisualization.cvt_imgarray2image(filters, margin=2)
                out_filters.append(filters)
                out_imgs.append(img_e)
        else:
            filters = np.array([f[0][0] for f in kept_filters])
            img_s, filters = MyNetVisualization.cvt_imgarray2image(filters, margin=2)
            out_filters.append(filters)
            out_imgs.append(img_s)

        if is_show:
            r, c = MyNetVisualization.find_best_row_col(len(out_imgs))
            plt.figure(num='Convolution kernel of \'' + layer_name + '\'')
            for i in range(len(out_imgs)):
                plt.subplot(r, c, i + 1)
                plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
                plt.axis('off')
                plt.title(input_layer_names[i])
                plt.imshow(out_imgs[i])
            plt.show()

        return out_imgs, out_filters

    @staticmethod
    def show_heatmap(model, last_conv_layer, img_path, preprcess_fun, is_show=True):
        """

        :param model:
        :param last_conv_layer:
        :param img_path:
        :param preprcess_fun:
        :param is_show:
        :return:
        """
        model_input = preprcess_fun(img_path)
        # predict the image class
        preds = model.predict(model_input)
        # find the class index
        index = np.argmax(preds[0])
        # This is the entry in the prediction vector
        target_output = model.output[:, index]

        # get the last layer
        last_conv_layer = model.get_layer(last_conv_layer)

        # compute the gradient of the output feature map with this target class
        grads = keras.backend.gradients(target_output, last_conv_layer.output)[0]

        # mean the gradient over a specific feature map channel
        pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))

        # this function returns the output of last layer and grads
        # given the input picture
        iterate = keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([model_input])

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the target class

        for i in range(conv_layer_output_value.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # for show result
        # We use cv2 to load the original image
        img_c = cv2.imread(img_path)

        # We resize the heatmap to have the same size as the original image
        heatmap = cv2.resize(heatmap, (img_c.shape[1], img_c.shape[0]))

        # We convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img_c
        superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

        if is_show:
            plt.figure(num='Heatmap and blend image')

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
            plt.title('Heat map')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
            plt.title('Blend image')

            plt.show()

        return heatmap, superimposed_img


# from keras.applications import VGG16, InceptionResNetV2
# from my_data_process import MyDataProcess
# from keras.preprocessing import image
# from imagenet_utils import preprocess_input, decode_predictions
#
# from my_data_process import MyDataProcess

# from MEDnet_NP import jaccards_loss

# if __name__ == '__main__':
# model = VGG16()
# model.summary()
# img_path = r'C:\Users\Yukun\Documents\Nutstore\i128.jpg'

# features = MyNetVisualization.show_featuremaps(model,'block3_conv2',img_path, MyDataProcess.imgnet_read_image,
# show_index_range=(0,16))

# MyNetVisualization.show_conv_kernel(model=model, layer_name='block4_conv3', model_input_shape=(224,224,3),
#                                                input_layer_names=None, show_index_range=(0,8), verbose=2,
# iteration=200)

# heatmap = MyNetVisualization.show_heatmap(model,'block5_conv3',img_path, MyDataProcess.imgnet_read_image)

# model_path = r'D:\Yukun\workspace\Shadow_NPA_Version_6\Code\Python\NPA_detection_CNN\logs\model_0_192_5.3282.hdf5'
# model = MyNetVisualization.load_model(model_path, custom_objects={'jaccards_loss': jaccards_loss})

# model = MyNetVisualization.load_model(r'D:\Yukun\workspace\EPS_Detection_CNN\logs0\model_Fluid_001_0.819.hdf5')
models = MyNetVisualization.load_model(r'D:\Yukun\workspace\EPS_Detection_CNN\logs\model_Fluid_499_0.029.hdf5')
img, kernels = MyNetVisualization.show_conv_kernel(model=models, layer_name='activation_20',
                                                   show_index_range=(0, 7),
                                                   input_layer_names=None, is_show=True, iteration=20)
