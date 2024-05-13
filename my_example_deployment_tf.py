import os
from template_cnn import ExampleCNN
import tensorflow as tf
import keras
import tf2onnx
import numpy as np
import onnxruntime as ort

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# parameters
nclass = 13
# create model
model = ExampleCNN(input_shape=(None,None,1), n_class=nclass)
model.load_model()

# deploy options
# 1. tf.saved_model
# 2. onnx
# 3. tfjs
# 4. tflite

deploy_option = 2

# load model
model = ExampleCNN(input_shape=(None,None,1), n_class=13)
model.load_model()

if deploy_option==1:
    # Option 1. convert model to tensorflow saved model
    tf.saved_model.save(model.model, './model/tf_saved_model')
    # test the saved model
    tfmodel = keras.layers.TFSMLayer('./model/tf_saved_model', call_endpoint='serving_default')
    y = tfmodel(tf.random.uniform((1, 480, 288, 1)))
    print('Test: tf.saved_model: ',tf.reduce_any(tf.equal(y['output_1'].shape,tf.constant([1, 480, 288, 13])) & tf.equal(y['output_0'].shape,tf.constant([1, 480, 288, 1]))).numpy())

if deploy_option==2:
    # Option 2. convert model to onnx
    spec = (tf.TensorSpec((None,None, None, 1), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model.model, input_signature=spec, opset=13, output_path="model/onnx/model.onnx")
    output_names = [n.name for n in model_proto.graph.output]
    ort_sess = ort.InferenceSession('model/onnx/model.onnx')
    y = ort_sess.run(None, {'input': np.random.rand(1, 480, 288, 1).astype(np.float32)})
    print('Test: onnx: ',np.equal(y[0].shape,np.array([1, 480, 288, 13])))

# if deploy_option==3: # not working for keras 3
#     # Option 3. convert tf model to tfjs
#     import tensorflowjs as tfjs
#     tfjs.converters.save_keras_model(model.model, './model/tfjs/tfjs_model')

# if deploy_option==4: # not working for keras 3
#     # Option 4. convert model to tflite
#     converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
#     tflite_model = converter.convert()
#     open("model.tflite", "wb").write(tflite_model)
#     # Load TFLite model and allocate tensors.
#     interpreter = tf.lite.Interpreter(model_content=tflite_model)
#     interpreter.allocate_tensors()
#     # Get input and output tensors.
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     print(input_details)
#     print(output_details)

#### The following code for Matlab(R2024a) load the onnx model

# net = importNetworkFromONNX("Z:\GitRepositories\Keras-Template\model\onnx\model.onnx");
# tic
# net = removeLayers(net,'input');
# net = removeLayers(net,'Unsqueeze_To_ReshapeLayer1234');
# inputlayer = imageInputLayer([304,304,1],Name='input',Normalization='none');
# net = addLayers(net,inputlayer);
# net = connectLayers(net,'input','Conv__1216');
# net = connectLayers(net,'input','Conv__1200');

# % remove  RegressionLayer_output0
# net = removeLayers(net,'maskOutput');
# net = removeLayers(net,'outOutput');
# net = removeLayers(net,'my_example_model__33');
# net = removeLayers(net,'Reshape_To_ReshapeLayer1254');
# net = connectLayers(net,'my_example_model__21','my_example_model__22');
# % sm = softmaxLayer('Name','sm');
# % net = addLayers(net,sm);
# % net = connectLayers(net,'x_out2_conv12_Conv','sm');
# % init network
# in = rand(304, 304,1,1);
# X = dlarray(in, 'SSCB');
# net = initialize(net, X);
# % test network
# out = minibatchpredict(net,img);

# [~,class] = max(out,[],3);
# figure(2),imshow(class,[]);
# toc