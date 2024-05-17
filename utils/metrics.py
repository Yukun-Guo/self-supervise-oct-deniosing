import keras
from keras import ops
import tensorflow as tf


def DiceScore():
    @keras.saving.register_keras_serializable()
    def dice(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred)
        dice_score = 2 * intersection / (union + 1e-8)
        return dice_score
    return dice
    

def JaccardCoefficient():
    @keras.saving.register_keras_serializable()
    def jaccard(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred) - intersection
        jaccard_coefficient = intersection / (union + 1e-8)
        return jaccard_coefficient
    return jaccard

def SSIM():
    @keras.saving.register_keras_serializable()
    def ssim(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        ssim_score = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return ssim_score
    return ssim

def PSNR():
    @keras.saving.register_keras_serializable()
    def psnr(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        psnr_score = tf.image.psnr(y_true, y_pred, max_val=1.0)
        return psnr_score
    return psnr