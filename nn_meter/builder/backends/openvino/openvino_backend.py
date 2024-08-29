# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, time

from ..interface import BaseBackend
from nn_meter.utils.path import get_filename_without_ext

from tensorflow import keras
from vpu_power_measurement import PowerCounter

class OpenVINOBackend(BaseBackend):
    parser_class = None
    profiler_class = None
    curr_folder_path = f"/media/locla/Data/tmp_data/workspace/predictor_build/power/save_2_0"
    
    def generate_folder(self, count):
        folder_id = int(count / 100)
        self.curr_folder_path = f"/media/locla/Data/tmp_data/workspace/predictor_build/power/save_2_{folder_id}"
        if not os.path.exists(self.curr_folder_path):
            os.mkdir(self.curr_folder_path)

    def update_configs(self):
        """update the config parameters for OpenVINO platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'venv': self.configs['OPENVINO_ENV'],
            'optimizer': self.configs['OPTIMIZER_PATH'],
            'runtime_dir': self.configs['OPENVINO_RUNTIME_DIR'],
            'serial': self.configs['DEVICE_SERIAL'],
            'data_type': self.configs['DATA_TYPE'],
        })
        self.venv = self.configs['OPENVINO_ENV']
    
    def convert_model(self, model_path, savedpath, input_shape=None):
        """convert the Keras model instance to frozen pb file
        """
        model_name = get_filename_without_ext(model_path)
        self.tmp_dir = os.path.join(savedpath, model_name)
        return model_path

    def profile(self, converted_model, metrics = ['latency'], input_shape = None):
        """convert the model to the backend platform and run the model on the backend, return required metrics 
        of the running results. We only support latency for metric by now.
        """
        self.profiler.load_graph(converted_model, self.tmp_dir)
        return self.profiler.profile(input_shape) # self.parser.parse).results.get(metrics)
    
    def backup_logs(self, count):
        # print("WAIT FOR BACKING UP")
        c = PowerCounter(dummy=False)
        # list_files = c.get_all_log_files()
        start= time.time()
        result = c.save_all_log_file_to_local(self.curr_folder_path, self.profiler.list_files)
        self.profiler.list_files = []
        self.generate_folder(count)
        print("BACKING UP: ", time.time() - start)
        return result

    def reset_instrument(self):
        c = PowerCounter(dummy=False)
        c.reset_multi_meter()