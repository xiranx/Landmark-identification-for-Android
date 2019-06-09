from keras.models import Model
from keras.layers import *
import os
import tensorflow as tf
from keras.models import model_from_json
from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input

def keras_to_tensorflow(keras_model,output_dir,model_name,out_prefix='output_',log_tensorboard=True):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i+1))
        tf.identity(keras_model.output[i],out_prefix + str(i + 1))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name),output_dir)

with open('resnet_model.json','r') as file:
    json_string = file.read()

keras_model = model_from_json(json_string)
keras_model.load_weights('parameters/resnet-e02-acc1.00.hdf5')

output_dir = os.path.join(os.getcwd(),'tensorflow model')

keras_to_tensorflow(keras_model,output_dir=output_dir,model_name='new_resnet.pb')

print('Model Saved')