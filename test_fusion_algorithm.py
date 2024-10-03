# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.kernel_detector.kernel_detector import KernelDetector

from nn_meter.ir_converter import model_file_to_graph, model_to_graph

rule_file = "/media/locla/Data/tmp_data/workspace/predictor_build/results/detected_fusion_rule_new_gpu.json"
kernel_detector = KernelDetector(rule_file, useParallel=True)

graph = model_file_to_graph("/home/locla/tmp_data/benchmark_models/Inception_V3/model.onnx",
                            "onnx", input_shape=(1, 3, 224, 224))
kernel_detector.load_graph(graph)
kernels = kernel_detector.get_kernels()
for kernel in kernels:
    if kernel['op'] == 'conv-relu':
        print(kernel)
