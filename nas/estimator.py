
import logging
from .utils import adjust_model_code

from nni.retiarii.converter import convert_to_graph
from nni.retiarii.converter.graph_gen import GraphConverterWithShape
from nni.retiarii.converter.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node

from nn_meter import load_predictor
import torch, nni, onnx, os
import inspect, re
_logger = logging.getLogger(__name__)


class HardwareMetricEstimator:
    def __init__(self, applied_hardware, hardware_metrics=["latency"]):
        import nn_meter  # pylint: disable=import-error
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
        self.predictor = nn_meter.load_predictor(applied_hardware, hardware_metrics)

    def estimate(self, model, dummy_input=(1, 1, 33)):
        path = inspect.getfile(model.__class__)
        dir_path = path.split("/")[-2]
        file_name = path.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{dir_path}/{file_name}.onnx"):
            torch.onnx.export(model, torch.randn(*dummy_input), f"{dir_path}/{file_name}.onnx")
        # print("test", f"{dir_path}/{file_name}.onnx")
        result = self.predictor.predict(f"{dir_path}/{file_name}.onnx", model_type="onnx") ## 'onnx , model_type = 'nni-ir'
        return result

    
@nni.trace
class HardwareMetricFilter:
    def __init__(self, thresholds, applied_hardware, reverse=False):
        """
        Filter the models according to predcted latency.
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
            
        self.predictors = load_predictor(applied_hardware)
        self.thresholds = thresholds
        self.reverse = reverse
        
    def __call__(self, ir_model):
        estimated_values = self.predictors.predict(ir_model, model_type = 'nnmeter-ir')# nni-ir
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
