# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, subprocess, time, json, sys, signal
from ..interface import BaseProfiler
import numpy as np 

def extract_power(power_log_file):
    power_list = []
    with open(power_log_file, "r") as f:
        for line in f:
            power_list.append(float(line.split(",")[1]))
    if len(power_list) == 0:
        return 0

    top_ids = np.argsort(power_list)[-10:]
    top_values = [power_list[i] for i in top_ids]

    return sum(top_values)/len(top_values) # 


class TRTProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, data_type):
        """
        @params:
        """
        if data_type == "FP16":
            self._data_type = "--fp16" 
        elif data_type == "BEST":
            self._data_type = "--best" 
        elif data_type == "INT8":
            self._data_type = "--int8"
        else:
            raise f"DATA TYPE IS INVALID: {data_type}"

    def load_model(self, model_file, tmp_dir):
        self._model_path = model_file
        self.tmp_dir = tmp_dir

    def profile(self, mode="jetson", profilePower=True, buildEngine=True):
        tmp_dir = "/".join(self._model_path.split("/")[:-1])
        new_tmp_dir = self._model_path.replace(".onnx", "")
        if not os.path.exists(new_tmp_dir):
            os.mkdir(new_tmp_dir)
        # build_command = f'/usr/src/tensorrt/bin/trtexec --buildOnly --onnx={os.path.join(tmp_dir,"model.onnx")} --saveEngine={os.path.join(tmp_dir,"model_engine.plan")}'
        build_command = f'/usr/src/tensorrt/bin/trtexec --buildOnly --onnx={self._model_path} --saveEngine={os.path.join(new_tmp_dir,"model_engine.plan")}'
        
        if profilePower:
            warmUpTime = 0
            runningTime = 5
        else:
            warmUpTime = 30
            runningTime = 10
        retry = 1
        profiling_command = f'sudo python3 run_profiling.py --engine_file {os.path.join(new_tmp_dir, "model_engine.plan")} \
                                                            --warm_up {warmUpTime} \
                                                            --running_time {runningTime} \
                                                            --exported_folder {new_tmp_dir} \
                                                            --mode {mode} \
                                                            --profilePower {profilePower}'
        # print("BUILD: ", build_command)
        # print("PROFILING: ", profiling_command)
        while True:
            try:
                if buildEngine and not os.path.isfile(os.path.join(new_tmp_dir,"model_engine.plan")):
                    subprocess.run(build_command, shell=True, timeout=1200, stdout=open(os.path.join(new_tmp_dir, 'profiled_log.txt'), 'w'))
                
                # time.sleep(1)
                if not os.path.isfile(os.path.join(new_tmp_dir,"model_engine.plan")):
                    raise Exception(f"Not exists file: {os.path.join(new_tmp_dir,'model_engine.plan')}")
                    
                # print (f"[RUNNING]: {profiling_command}")
                subprocess.run(
                        f'bash -c "{profiling_command}"',
                        shell=True,
                        timeout=1200, stdout=open(os.path.join(new_tmp_dir, 'profiled_log.txt'), 'w')
                )
                
                output = open(os.path.join(new_tmp_dir, 'latency.json'), 'r').read()
                
                break
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print (f"[FAIL]: {profiling_command}")
                print(exc_type, fname, exc_tb.tb_lineno)
                if retry == 0:
                    raise(e)

                print('Retrying...')
                retry -= 1
        output = json.loads(output)
        avg = output["all_layers"]["avg"]
        std = output["all_layers"]["std"]
        result = {
                "latency": f"{avg} +- {std}",
                "power_file": os.path.join(new_tmp_dir, "power_log.txt"),
                "power": extract_power(os.path.join(new_tmp_dir, "power_log.txt"))
            }

        return result