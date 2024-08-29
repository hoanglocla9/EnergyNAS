# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import shutil
import logging
from ..interface import BaseBackend
from nn_meter.utils.path import get_filename_without_ext
logging = logging.getLogger("nn-Meter")
import numpy as np 

class TFLiteBackend(BaseBackend):
    parser_class = None
    profiler_class = None

    def update_configs(self):
        """update the config parameters for TFLite platform
        """
        super().update_configs()
        # self.profiler_kwargs.update({
        #     'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
        #     'benchmark_model_path': self.configs['BENCHMARK_MODEL_PATH'],
        #     'serial': self.configs['DEVICE_SERIAL'],
        #     'dst_kernel_path': self.configs['KERNEL_PATH']
        # })

    def convert_model(self, model_path, save_path, input_shape = None):
        """convert the Keras model instance to ``.tflite`` and return model path
        """
        import tensorflow as tf
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if "reshape" not in model_path:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # We only want to use int8 kernels
            converter.inference_input_type = tf.uint8  # Can also be tf.int8
            converter.inference_output_type = tf.uint8   # Can also be tf.int8
            def representative_dataset_gen():
                for _ in range(100):
                    # Get sample input data as a numpy array in a method of your choosing.
                    if len(input_shape) >= 2 and len(input_shape[0]) > 0:
                        result = []
                        for s in input_shape:
                            s = [1, *s]
                            result.append(np.float32(np.random.randint(0, 255, size=s, dtype=np.uint8 )))
                        yield result
                    else:
                        s = [1, *(input_shape[0])]
                        yield [np.float32(np.random.randint(0, 255, size=s, dtype=np.uint8))]
            converter.representative_dataset = representative_dataset_gen

        tflite_model = converter.convert()
        converted_model = os.path.join(save_path, model_name + '.tflite')
        open(converted_model, 'wb').write(tflite_model)
        shutil.rmtree(model_path)
        return converted_model
        
    def profile(self, converted_model_path, metrics = ['latency'], input_shape = None):
        """convert the model to the backend platform and run the model on the backend, return required metrics 
        of the running results. We only support latency for metric by now.
        """
        self.profiler.load_model(converted_model_path)

        return self.profiler.profile(input_shape, metrics)