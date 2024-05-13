from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow.python.platform import gfile
from os import path

"""
Example:

from my_losses import my_jaccard_loss

keras_model_path = '../logs/model_01_0.23.hdf5'
custom_objects = {'my_jaccard_loss': my_jaccard_loss}
my_model_convert_keras2tensorflow(keras_model_path, custom_objects=custom_objects, out_dir=None, out_pb_filename=None)

"""


class my_model_tools(object):

    @staticmethod
    def my_model_convert_keras2tfv2(keras_model_path, out_dir, out_pb_filename, custom_objects=None, compile=False):
        model = keras.models.load_model(keras_model_path, custom_objects=custom_objects, compile=compile)
        model.save(path.join(out_dir, out_pb_filename))
        model.summary()

    @staticmethod
    def my_model_convert_keras2tfv1(keras_model_path, out_dir, out_pb_filename, custom_objects=None, compile=False,
                                    vabel=False):
        # Convert Keras model to ConcreteFunction
        network = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects, compile=compile)
        full_model = tf.function(lambda x: network(x))
        full_model = full_model.get_concrete_function(tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        if vabel:
            print("-" * 50)
            print("Frozen model layers: ")
            for layer in layers:
                print(layer)
                print("-" * 50)
            print("Frozen model inputs: ")
            print(frozen_func.inputs)
            print("Frozen model outputs: ")
            print(frozen_func.outputs)
        # Save frozen graph from frozen ConcreteFunction to hard drive
        if not out_pb_filename[-3:] == '.pb':
            out_pb_filename += '.pb'
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=out_dir, name=out_pb_filename, as_text=False)

    @staticmethod
    def my_model_convert_keras2tfjs(keras_model_path, out_dir, out_filename, custom_objects=None, compile=False):

        model = keras.models.load_model(keras_model_path, custom_objects=custom_objects, compile=False)
        # tfjs.converters.save_keras_model(model, path.join(out_dir,out_filename))
        model.summary()

