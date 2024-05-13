from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras import backend
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.keras import backend_config


def my_jaccard_loss_sigmoid(y_true, y_pred, smooth=1):
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    sum_ = backend.sum(backend.abs(y_true) + backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - jac


def my_jaccard_loss_softmax(num_classes=3, weights=None, label_smoothing=0., smooth=1):
    if not weights:
        w = backend.ones([num_classes, ], dtype='float32')
    else:
        w = ops.convert_to_tensor_v2(weights)
    labels = label_smoothing

    def _loss(y_true, y_pred):
        jac = 0
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_s = ops.convert_to_tensor_v2(labels, dtype=y_pred.dtype)
        wd = backend.cast(w, y_pred.dtype)

        def _smooth_labels():
            num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - label_s) + (label_s / num_classes)

        y_true = smart_cond.smart_cond(label_s, _smooth_labels, lambda: y_true)

        for idx in range(num_classes):
            jac += my_jaccard_loss_sigmoid(y_true[..., idx], y_pred[..., idx], smooth) * wd[idx]
        return jac

    return _loss


def my_dice_loss_sigmoid(y_true, y_pred, smooth=1):
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=-1)
    sum_ = backend.sum(backend.abs(y_true) + backend.abs(y_pred), axis=-1)
    dice = (2 * intersection + smooth) / (sum_ + smooth)
    return 1 - dice


def my_dice_loss_softmax(num_classes=2, weights=None, label_smoothing=0., smooth=1):
    if not weights:
        w = backend.ones([num_classes, ], dtype='float32')
    else:
        w = ops.convert_to_tensor_v2(weights)
    labels = label_smoothing

    def _loss(y_true, y_pred):
        dice = 0

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_s = ops.convert_to_tensor_v2(labels, dtype=y_pred.dtype)
        wd = backend.cast(w, y_pred.dtype)

        def _smooth_labels():
            num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - label_s) + (label_s / num_classes)

        y_true = smart_cond.smart_cond(label_s, _smooth_labels, lambda: y_true)

        for idx in range(num_classes):
            dice += my_dice_loss_sigmoid(y_true[..., idx], y_pred[..., idx], smooth) * wd[idx]
        return dice

    return _loss


def weighted_categorical_crossentropy(weights, label_smoothing=0.):
    """
    A weighted version of tf.keras.losses.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        y_true = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
        y_pred = tf.constant([.9, .05, .05, .5, .89, .6, .05, .01, .94], shape=[3,3])
        weights = [.1,.1,1.]
        loss_no_weight = tf.keras.losses.categorical_crossentropy(y_true,y_pred,label_smoothing=0.2)
        loss_weight = weighted_categorical_crossentropy(weights, label_smoothing=0.2)
        loss = loss_weight(y_true, y_pred)

        print(np.around(loss_no_weight, 5))
        print(np.around(loss, 5))

        >>[0.49074 0.8694  0.56035]
          [0.22882 0.15888 0.1043 ]

    """

    def _backtrack_identity(tensor):
        while tensor.op.type == 'Identity':
            tensor = tensor.op.inputs[0]
        return tensor

    def _constant_to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.

        This is slightly faster than the _to_tensor function, at the cost of
        handling fewer cases.

        Arguments:
            x: An object to be converted (numpy arrays, floats, ints and lists of
              them).
            dtype: The destination type.

        Returns:
            A tensor.
        """
        return constant_op.constant(x, dtype=dtype)

    weights = ops.convert_to_tensor_v2(weights)
    label_s = label_smoothing

    def _loss(y_true, y_pred):

        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        labels = ops.convert_to_tensor_v2(label_s, dtype=backend.floatx())

        def _smooth_labels():
            num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - labels) + (labels / num_classes)

        y_true = smart_cond.smart_cond(labels,
                                       _smooth_labels, lambda: y_true)

        y_true.shape.assert_is_compatible_with(y_pred.shape)
        if not isinstance(y_pred, (ops.EagerTensor, variables_module.Variable)):
            y_pred = _backtrack_identity(y_pred)
            if y_pred.op.type == 'Softmax':
                # When softmax activation function is used for y_pred operation, we
                # use logits from the softmax function directly to compute loss in order
                # to prevent collapsing zero when training.
                # See b/117284466
                assert len(y_pred.op.inputs) == 1
                y_pred = y_pred.op.inputs[0]
                return nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_true, logits=y_pred, axis=-1)

        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / math_ops.reduce_sum(y_pred, -1, True)
        # Compute cross entropy from probabilities.
        epsilon_ = _constant_to_tensor(backend_config.epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        return -math_ops.reduce_sum(y_true * math_ops.log(y_pred) * weights, -1)

    return _loss


def va_loss(y_true,y_pred, nclass=3):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true.shape.assert_is_compatible_with(y_pred.shape)

    y_pred = math_ops.argmax_v2(y_pred, -1)
    y_pred = tf.one_hot(y_pred,nclass)

    y_preds = tf.unstack(y_pred,axis=-1)
    y_trues = tf.unstack(y_true,axis=-1)
    loss = 0
    kernel = tf.zeros((3, 3, 1))
    for y_p, y_t in zip(y_preds,y_trues):
        y_p = tf.expand_dims(y_p, 3)
        y_p = tf.nn.erosion2d(y_p,kernel,strides=[1,1,1,1],padding='SAME',data_format="NHWC",dilations=[1,1,1,1])
        y_p = tf.nn.dilation2d(y_p,kernel,strides=[1,1,1,1],padding='SAME',data_format="NHWC",dilations=[1,1,1,1])
        y_p = tfa.image.connected_components(tf.squeeze(y_p, 3))

        y_t = tf.expand_dims(y_t, 3)
        y_t = tf.nn.erosion2d(y_t, kernel,strides=[1,1,1,1],padding='SAME',data_format="NHWC",dilations=[1,1,1,1])
        y_t = tf.nn.dilation2d(y_t, kernel,strides=[1,1,1,1],padding='SAME',data_format="NHWC",dilations=[1,1,1,1])
        y_t = tf.image.connected_components(tf.squeeze(y_t, 3))
        loss = loss+tf.reduce_mean(tf.abs(tf.reduce_max(y_p,[1,2])-tf.reduce_max(y_t,[1,2])))
    return loss

if __name__ == '__main__':
    # plot loss
    import matplotlib.pyplot as plt

    y_pred = tf.convert_to_tensor(np.array([np.arange(-1, 1 + 0.1, 0.01)]).T)
    y_true = tf.convert_to_tensor(np.zeros(y_pred.shape))

    loss_jaccard_sg = my_jaccard_loss_sigmoid(y_true, y_pred)
    loss_dic_sg = my_dice_loss_sigmoid(y_true, y_pred)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss_jaccard_sofx = my_jaccard_loss_softmax(num_classes=1, label_smoothing=0.2)(tf.expand_dims(y_true, 2),
                                                                                    tf.expand_dims(y_pred, 2))
    loss_dic_sofx = my_dice_loss_softmax(num_classes=1, label_smoothing=0.2)(tf.expand_dims(y_true, 2),
                                                                             tf.expand_dims(y_pred, 2))

    plt.subplot(3, 3, 1)
    plt.title('jaccard sgmiod')
    plt.plot(y_pred, loss_jaccard_sg)

    plt.subplot(3, 3, 2)
    plt.title('dice sgmiod')
    plt.plot(y_pred, loss_dic_sg)

    plt.subplot(3, 3, 3)
    plt.title('bce')
    plt.plot(y_pred, bce_loss)

    plt.subplot(3, 3, 4)
    plt.title('loss_jaccard_sofx')
    plt.plot(y_pred, loss_jaccard_sofx)

    plt.subplot(3, 3, 5)
    plt.title('loss_dic_sofx')
    plt.plot(y_pred, loss_dic_sofx)
    plt.show()

    # test loss
    y_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1., 0.]])
    y_pred = np.array([[0, 0, 0.9, 0], [0, 0, 0.1, 0], [1, 1, 0.1, 1.]])

    loss_jaccard_sg = my_jaccard_loss_sigmoid(y_true, y_pred)
    loss_dic_sg = my_dice_loss_sigmoid(y_true, y_pred)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss_jaccard_sofx = my_jaccard_loss_softmax(num_classes=1, label_smoothing=0.2)(tf.expand_dims(y_true, 2),
                                                                                    tf.expand_dims(y_pred, 2))
    loss_dic_sofx = my_dice_loss_softmax(num_classes=1, label_smoothing=0.2)(tf.expand_dims(y_true, 2),
                                                                             tf.expand_dims(y_pred, 2))

    print(loss_jaccard_sg)
    print(loss_jaccard_sofx)
    print(loss_dic_sg)
    print(loss_dic_sofx)
