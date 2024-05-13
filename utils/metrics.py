import keras
import keras.ops as ops


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

