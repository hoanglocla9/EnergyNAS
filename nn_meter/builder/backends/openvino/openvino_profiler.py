# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import subprocess
import numpy as np
import shutil
import pandas
import csv, time

from .utils import restart
from ..interface import BaseProfiler
from nn_meter.utils.pyutils import get_pyver
from openvino.tools.benchmark.benchmark import Benchmark
from datetime import datetime
from vpu_power_measurement import start_power_reading_v2, stop_power_reading_v2

from threading import Thread
from openvino_benchmark.main import main
import string, random
from rpc import RPCClient


class OpenVINOProfiler(BaseProfiler):
    server = RPCClient('0.0.0.0', 8080)
    server.connect()

    device = None
    def __init__(self, venv, optimizer, runtime_dir, serial, graph_path='', _dst_graph_path='', data_type='FP16'):
        self._graph_path = graph_path
        self._venv = venv
        self._optimizer = optimizer
        self._dst_graph_path = _dst_graph_path
        self._runtime_dir = runtime_dir
        self._serial = serial
        self._data_type = data_type
        self.count = 0
        self.list_files = []

    def load_graph(self, graph_path, dst_graph_path):
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path

    def profile(self, shapes, retry=2):
        self.count+= 1
        filename = os.path.splitext(os.path.basename(self._graph_path))[0]
        input_path = os.path.join(self._dst_graph_path, 'inputs')

        if not os.path.exists(os.path.join(self._dst_graph_path, filename + ".xml")):
            try:
                subprocess.run(
                    f'mo '
                    f'--input_model {self._graph_path} '
                    f'--output_dir {self._dst_graph_path} ' 
                    f'--log_level ERROR '
                    f'-b 1 '
                    f'--data_type {self._data_type} ' ,
                    shell=True  
                )
            except Exception as e:
                print("ERROR", e)
                return {"latency": -1}

            if os.path.exists(input_path):
                shutil.rmtree(input_path)

            os.mkdir(input_path)
            for index, shape in enumerate(shapes):
                input = np.random.rand(*shape).astype(np.float32).tofile(os.path.join(input_path, f'input_{index}.bin'))
        running_time = 3
        while True:
            try:
                server.profilingPower(os.path.join(self._dst_graph_path, 'power_log.txt'), running_time+1)
                main([input_path], os.path.join(self._dst_graph_path, filename + ".xml"), self._dst_graph_path, running_time)
                df = pandas.read_csv(os.path.join(self._dst_graph_path, 'benchmark_detailed_counters_report.csv'), delimiter=";")
                df.reset_index()
                extra_latency = 0
                for index, row in df.iterrows():
                    if row['layerType'] == "<Extra>":
                        extra_latency += row["realTime (ms)"]
                latency = df.iloc[-1]["realTime (ms)"] - extra_latency
                # if latency < 1.2 and current_iter == 50:
                #     current_iter = int(5/latency * 50)
                #     continue
                break
            except Exception as e:
                if retry == 0:
                    print("ERROR", e)
                    return {"latency": -1}
                retry -= 1

        if os.path.exists(os.path.join(self._dst_graph_path, 'power_log.txt')):
            output = {"latency": latency, "power_file": os.path.join(self._dst_graph_path, 'power_log.txt')}
        else:
            print("SOME THING WRONG WITH POWER SERVER!!!!")
            output = {"latency": latency}
        # if power_file != None and power_file != "":
        #     output = {"latency": latency, "power_file": power_file}
        #     print("OK ", input_path, output)
        #     self.list_files.append(power_file)
        # else:
        #     print(f"ERROR with power_file at {input_path}")
        #     output = {"latency": -1, "power_file": ""}
        return output
