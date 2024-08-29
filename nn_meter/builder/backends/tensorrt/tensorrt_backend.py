# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import shutil
import logging
from ..interface import BaseBackend
from nn_meter.utils.path import get_filename_without_ext
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
logging = logging.getLogger("nn-Meter")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

class TRTBackend(BaseBackend):
    parser_class = None
    profiler_class = None
    tmp_dir = ""
    def update_configs(self):
        """update the config parameters for TFLite platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'data_type': self.configs['DATA_TYPE']
        })

    def convert_model(self, model_path, save_path, input_shape=None):
        """convert the Keras model instance to ``.tflite`` and return model path
        """
        import tensorflow as tf
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        import tf2onnx, onnx
        # onnx_model = keras2onnx.convert_keras(model, model.name)
        model_tmp_dir = os.path.join(save_path, model_name)
        if not os.path.exists(model_path + "/tmp"):
            os.makedirs(model_path + "/tmp")
        self.tmp_dir = model_path + "/tmp"
        # model_file = model_tmp_dir + "/" +"model.onnx"
        # keras2onnx.save_model(onnx_model, model_file)

        onnx_model, _ = tf2onnx.convert.from_keras(model, inputs_as_nchw=["input_1", "inputs"], outputs_as_nchw=["output_1", "outputs"], opset=13)
        onnx.save(onnx_model, model_path+"/model.onnx")              
        return model_path+"/model.onnx"
        
    def profile(self, converted_model_path, tmp_dir, metrics = ['latency'], input_shape = None, mode="jetson"):
        """convert the model to the backend platform and run the model on the backend, return required metrics 
        of the running results. We only support latency for metric by now.
        """
        self.profiler.load_model(converted_model_path, tmp_dir)

        return self.profiler.profile(mode)