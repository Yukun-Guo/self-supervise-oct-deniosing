
from keras import ops
import keras
AvaliableLosses = {
    'MeanSquareErrorLoss',
    'DiceLoss',
    'JaccardLoss',
    'FocalLoss',
    'TverskyLoss',
    'BinaryCrossEntropyLoss',
    'CategoricalCrossEntropyLoss',
    'FocalTverskyLoss'
}

def MeanSquareErrorLoss():
    @keras.saving.register_keras_serializable()
    def mse(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        mse = ops.mean(ops.square(y_true - y_pred), axis=-1)
        return mse
    return mse

def DiceLoss(smooth=1):
    @keras.saving.register_keras_serializable()
    def dice_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred, axis=-1)
        union = ops.sum(y_true, axis=-1) + ops.sum(y_pred, axis=-1)
        dice = (2. * intersection + smooth) / (union + smooth)
        return ops.mean(1 - dice)
    return dice_loss
    
def JaccardLoss(smooth=1):
    @keras.saving.register_keras_serializable()
    def jaccard_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred) - intersection
        jaccard = (intersection + smooth) / (union + smooth)
        return 1 - jaccard
    return jaccard_loss

def FocalLoss(gamma=2.0, alpha=0.25):
    @keras.saving.register_keras_serializable()
    def focal_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = - alpha * y_true * ops.power(1 - y_pred, gamma) * np.log(y_pred) \
               - (1 - alpha) * (1 - y_true) * ops.power(y_pred, gamma) * np.log(1 - y_pred)
        return ops.mean(loss)
    return focal_loss
    
def TverskyLoss(smooth=1, alpha=0.7):
    @keras.saving.register_keras_serializable()
    def tversky_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred)
        fps = ops.sum((1 - y_true) * y_pred)
        fns = ops.sum(y_true * (1 - y_pred))
        tversky = (intersection + smooth) / (intersection + alpha * fps + (1 - alpha) * fns + smooth)
        return 1 - tversky
    return tversky_loss

def BinaryCrossEntropyLoss(epsilon=1e-7):
    @keras.saving.register_keras_serializable()
    def bce_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        y_pred = ops.clip(y_pred, epsilon, 1 - epsilon)
        bce_loss = - (y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))    
        return ops.mean(bce_loss)
    return bce_loss

def CategoricalCrossEntropyLoss(epsilon=1e-7):
    @keras.saving.register_keras_serializable()
    def cce_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        y_pred = y_pred / ops.sum(y_pred,axis=-1, keepdims=True)
        y_pred = ops.clip(y_pred, epsilon, 1 - epsilon)
        cce_loss = - ops.sum(y_true * ops.log(y_pred),axis=-1)
        return ops.mean(cce_loss)  # Normalize by the number of samples
    return cce_loss

def FocalTverskyLoss(smooth=1, gamma=2.0, alpha=0.7):
    @keras.saving.register_keras_serializable()
    def focal_tversky_loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)
        intersection = ops.sum(y_true * y_pred)
        fps = ops.sum((1 - y_true) * y_pred)
        fns = ops.sum(y_true * (1 - y_pred))
        tversky = (intersection + smooth) / (intersection + alpha * fps + (1 - alpha) * fns + smooth)
        loss = ops.power((1 - tversky), gamma)
        return ops.mean(loss)
    return focal_tversky_loss


def CombinedLoss(losses, weights=None,**kwargs):
    def _parse_losses( losses, **kwargs):
        parsed_losses = []

        for loss in losses:
            if isinstance(loss, str):
                parsed_losses.append(_get_loss_instance(loss, **kwargs.get(loss, {})))
            elif callable(loss):
                parsed_losses.append(loss)
            else:
                raise ValueError(f"Unsupported loss format: {loss}")
        return parsed_losses

    def _get_loss_instance(loss_name, **kwargs):
        available_losses = {
            'DiceLoss': DiceLoss,
            'JaccardLoss': JaccardLoss,
            'FocalLoss': FocalLoss,
            'TverskyLoss': TverskyLoss,
            'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
            'CategoricalCrossEntropyLoss': CategoricalCrossEntropyLoss,
            'FocalTverskyLoss': FocalTverskyLoss,
        }
        if loss_name not in available_losses:
            raise ValueError(f"Unsupported loss: {loss_name}")
        return available_losses[loss_name](**kwargs)
    losses = _parse_losses(losses, **kwargs)
    if weights is None:
        weights = [1] * len(losses)
    else:
        weights = weights

    if len(losses) != len(weights):
        raise ValueError("Number of losses and weights must be the same.")

    @keras.saving.register_keras_serializable()
    def comb_loss(y_true, y_pred):
        total_loss = 0
        for loss, weight in zip(losses, weights):
            total_loss += weight * loss(y_true, y_pred)
        return total_loss   
    return comb_loss
    
if __name__ == '__main__':
    import numpy as np
    import keras
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.1], [0.6, 0.5, 0.2]])
    bce_loss = BinaryCrossEntropyLoss()(y_true, y_pred)
    bce_loss_k = keras.losses.BinaryCrossentropy()(y_true, y_pred)
    cce_loss = CategoricalCrossEntropyLoss()(y_true, y_pred)
    cce_loss_k = keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    dice_loss = DiceLoss()(y_true, y_pred)
    jaccard_loss = JaccardLoss()(y_true, y_pred)
    focal_loss = FocalLoss()(y_true, y_pred)
    tversky_loss = TverskyLoss()(y_true, y_pred)
    focal_tversky_loss = FocalTverskyLoss()(y_true, y_pred)

    combined_loss = CombinedLoss(['DiceLoss', 'JaccardLoss'], [0.5, 0.5])(y_true, y_pred)
    print("bce_loss",bce_loss)
    print("bce_loss_k",bce_loss_k)
    print("cce_loss",cce_loss)
    print("cce_loss_k",cce_loss_k)
    print("dice_loss",dice_loss)
    print("jaccard_loss",jaccard_loss)
    print("focal_loss",focal_loss)
    print("tversky_loss",tversky_loss)
    print("focal_tversky_loss",focal_tversky_loss)
    print("combined_loss",combined_loss)
    print('DiceLoss+JaccardLoss',0.5*dice_loss+0.5*jaccard_loss)



