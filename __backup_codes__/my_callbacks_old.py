"""
version 1.1
Yukun Guo
CEI, OHSU
"""
import pickle
import os
import io
import six
import csv
import numpy as np
from PIL import Image
from collections import OrderedDict, Iterable

import tensorflow as tf
from keras import backend
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, CSVLogger

class MyTensorBoard(TensorBoard):

    def __init__(self,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 write_leanrning_rate=True,
                 write_layer_outputs='all',
                 validation_data=None,
                 validation_batch_number=1):
        super().__init__(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         write_graph=write_graph,
                         write_images=write_images,
                         update_freq=update_freq,
                         profile_batch=profile_batch,
                         embeddings_freq=embeddings_freq,
                         embeddings_metadata=embeddings_metadata)
        self.write_lr = write_leanrning_rate
        self.write_layer_outputs = write_layer_outputs
        self.validation_data = validation_data
        self.validation_batch_number = validation_batch_number

    def on_epoch_end(self, epoch, logs=None):
        if self.write_lr:
            logs.update({'lr': backend.eval(self.model.optimizer.lr)})
        if self.write_layer_outputs is not None and self.validation_data is not None and self.validation_batch_number > 0:
            inp = self.model.input
            outputs = ()
            outputs_name = ()
            all_layer_names = [layer.name for layer in self.model.layers]
            data = self.validation_data.take(self.validation_batch_number)
            if isinstance(self.write_layer_outputs, str) and self.write_layer_outputs.lower() == 'all':
                outputs = [layer.output for layer in self.model.layers]
                outputs_name = [layer.name for layer in self.model.layers]
            elif isinstance(self.write_layer_outputs, tuple) or isinstance(self.write_layer_outputs, list):
                outputs = [self.model.get_layer(ly).output for ly in self.write_layer_outputs if ly in all_layer_names]
                outputs_name = [self.model.get_layer(ly).name for ly in self.write_layer_outputs if
                                ly in all_layer_names]
            elif isinstance(self.write_layer_outputs, str):
                if self.write_layer_outputs in all_layer_names:
                    outputs = [self.model.get_layer(self.write_layer_outputs).output]
                    outputs_name = [self.model.get_layer(self.write_layer_outputs).name]
            if outputs is not None:
                layer_outs = tf.keras.Model(inp, outputs).predict(data)
                if not isinstance(layer_outs, list):
                    layer_outs = [layer_outs]

                with self._get_writer(self._train_run_name).as_default():
                    for out_name, outp in zip(outputs_name, layer_outs):
                        imgs = [np.expand_dims(outp[:, :, :, i], 3) for i in range(np.size(outp, 3))]
                        for ct, img in enumerate(imgs):
                            tf.summary.image(name=out_name + '\outputs:chn#' + str(ct), data=img, step=epoch)
                    for img, lbl in data:
                        tf.summary.image(name='input', data=img, step=epoch)
                        tf.summary.image(name='label', data=lbl, step=epoch)
                        break
        super().on_epoch_end(epoch, logs)


class MyCSVLogger(CSVLogger):

    def __init__(self, log_dir=r'.\logs', write_lr=1, write_layer_name=True, append=True, *args, **kwargs):
        self.write_lr = write_lr
        self.logdir = log_dir
        self.write_layer_name = write_layer_name
        super().__init__(os.path.join(log_dir, 'training_log.csv'), append=append, *args, **kwargs)

    def on_train_begin(self, logs=None):
        if self.write_layer_name:
            infofp = open(os.path.join(self.logdir, 'layer_names.info'), 'w')
            for ly in self.model.layers:
                infofp.write(ly.name + '\n')
            infofp.close()
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.write_lr:
            logs.update({'lr': backend.eval(self.model.optimizer.lr)})

        epoch += 1
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0

            def _format_value(k):
                return '{:<10}'.format(k)

            if isinstance(k, six.string_types):
                return '{:<5}'.format(k)
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(_format_value, k)))
            else:
                return '{:0<2.9f}'.format(k)

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': '{:>5}'.format(epoch)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class MyModelCheckpoint(ModelCheckpoint):
    """
    Example:
        MyModelCheckpoint(save_callbacks={(lr_callbk,'lr_callbk_filename')})

    """

    def __init__(self, filepath, save_lr=1, checkpoint_file='checkpoint.ckp', save_best_only=False, *args, **kwargs):
        # Added arguments
        self.save_lr = save_lr
        self.checkpoint_file = checkpoint_file
        if save_best_only:
            filepath = os.path.join(os.path.dirname(filepath), 'best_model{}'.format(os.path.splitext(filepath)[1]))
        self.filepaths = filepath
        super().__init__(filepath, save_best_only=save_best_only, *args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):

        super().on_epoch_end(epoch, logs)
        pathf = self._get_file_path(epoch, logs)
        fn = os.path.join(os.path.dirname(pathf), self.checkpoint_file)

        md_fn = os.path.basename(pathf)
        lr = backend.get_value(self.model.optimizer.lr)

        if self.epochs_since_last_save == 0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current <= self.best:
                    # Note, there might be some cases where the last statement will save on unwanted epochs.
                    # However, in the usual case where your monitoring value space is continuous this is not likely
                    if self.save_lr:
                        epoch_s = '{} '.format(epoch + 1)
                        lr_s = '{:.0e} '.format(lr)
                        with open(fn, "w") as f:
                            f.write(epoch_s)
                            f.write(lr_s)
                            f.write(md_fn)
            else:
                if self.save_lr:
                    epoch_s = '{} '.format(epoch + 1)
                    lr_s = '{:.0e} '.format(lr)
                    with open(fn, "w") as f:
                        f.write(epoch_s)
                        f.write(lr_s)
                        f.write(md_fn)


class MySetLearningRate(Callback):
    def __init__(self, init_lr=0.0003, *args, **kwargs):
        self.init_lr = init_lr
        super().__init__()

    def on_train_batch_begin(self, batch, logs=None):
        if batch == 0:
            backend.set_value(self.model.optimizer.lr, self.init_lr)
