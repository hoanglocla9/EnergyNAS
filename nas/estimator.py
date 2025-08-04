import logging
from .utils import adjust_model_code, extract_model_layers_sizes

from nni.retiarii.converter import convert_to_graph
from nni.retiarii.converter.graph_gen import GraphConverterWithShape
from nni.retiarii.converter.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node

# from nn_meter import load_predictor
import torch
import nni
import onnx
import os
import inspect
import re

from prediction_function import InferencePredictor
_logger = logging.getLogger(__name__)


@nni.trace
class HardwareMetricEstimator:
    def __init__(self, applied_hardware, hardware_metrics=["latency"]):
        # import nn_meter  # pylint: disable=import-error
        _logger.info(
            f'Load latency predictor for applied hardware: {applied_hardware}.')
        # self.predictor_name = applied_hardware
        # self.predictor = nn_meter.load_predictor(
        #     applied_hardware, hardware_metrics) #edit here to add own predictors.
        self.predictor = InferencePredictor()

    def estimate(self, model, dummy_input=(1, 33)):
        path = inspect.getfile(model.__class__)
        dir_path = path.split("/")[-2]
        file_name = path.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{dir_path}/{file_name}.onnx"):
            torch.onnx.export(model, torch.randn(*dummy_input),
                              f"{dir_path}/{file_name}.onnx")
        
        # print("test", f"{dir_path}/{file_name}.onnx")
        # 'onnx , model_type = 'nni-ir'
        # result = self.predictor.predict(
        #     f"{dir_path}/{file_name}.onnx", model_type="onnx")
        result = 0
        layers = extract_model_layers_sizes(f"{dir_path}/{file_name}.onnx")
        for layer in layers:
            i, o = layer
            result += self.predictor.predict(i, o)
            result += 1.95
        return result


@nni.trace
class HardwareMetricFilter:
    def __init__(self, thresholds, applied_hardware, reverse=False):
        """
        Filter the models according to predicted latency.
        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwisse
        """

        self.predictors = InferencePredictor()
        self.thresholds = thresholds
        self.reverse = reverse

    def __call__(self, model_path: str):
        estimated_values = []
        for i, o in extract_model_layers_sizes(model_path):
            estimated_values.append(self.predictors.predict(model_path))
        estimated_values = tuple(estimated_values)
        if not self.reverse:
            result = True
            for m in self.thresholds:
                result = result and estimated_values[m] < self.thresholds[m]
            return result
        else:
            result = True
            for m in self.thresholds:
                result = result and estimated_values[m] > self.thresholds[m]
            return result
