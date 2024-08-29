# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, time
from ..interface import BaseProfiler

from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import numpy as np 
from PIL import Image
import statistics


class TFLiteProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, graph_path='', num_runs=100, warm_ups=50):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        """
        self._graph_path = graph_path
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def load_model(self, graph_path):
        self._graph_path = graph_path
        self.interpreter = make_interpreter(self._graph_path)

    def profile(self, shapes, metrics):
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        # Model must be uint8 quantized
        
        if common.input_details(self.interpreter, 'dtype') == np.uint8:
            # raise ValueError('Only support uint8 input type.')
        # if useUINT8:
            if len(shapes) >= 2 and len(shapes[0]) > 0:
                result = []
                for idx, s in enumerate(shapes):
                    s = [1, *s]
                    input_text = np.random.randint(0, 255, size=s, dtype=np.uint8)
                    self.interpreter.set_tensor(input_details[idx]['index'], input_text)
            else:
                s = [1, *(shapes[0])]
                input_text =np.random.randint(0, 255, size=s, dtype=np.uint8)
                self.interpreter.set_tensor(input_details[0]['index'], input_text)
        else:
            s = [1, *(shapes[0])]
            input_text =np.float32(np.random.randint(0, 255, size=s, dtype=np.uint8))
            self.interpreter.set_tensor(input_details[0]['index'], input_text)
            
        for i in range(self._warm_ups):
            self.interpreter.invoke()
        latencies = []
        for i in range(self._num_runs):
            start = time.perf_counter()
            self.interpreter.invoke()
            latency = time.perf_counter() - start
            latencies.append(latency*1000)
        mean_latency = sum(latencies) / len(latencies)
        std_latency = statistics.stdev(latencies)
        # print("std", std_latency)
        if std_latency == 0:
            print(latencies)
        output = {"latency": f"{mean_latency} +- {std_latency}", "power_file": ""}
        return output
