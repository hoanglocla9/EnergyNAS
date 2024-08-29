import os, math, nni, sys, torch, argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pandas import DataFrame, concat
from collections import OrderedDict
import nni.retiarii.strategy as strategy
from nn_meter import load_latency_predictor
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from model import *
from mutator import *
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.converter.graph_gen import GraphConverterWithShape
from nni.retiarii.converter.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node
import logging
import warnings
import inspect

os.environ["PICKLE_SIZE_LIMIT"] = str(64*1024*1024)
_logger = logging.getLogger(__name__)



class HardwareLatencyEstimator:
    def __init__(self, applied_hardware):
        import nn_meter  # pylint: disable=import-error
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
        self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)

    def estimate(self, model, dummy_input=(1, 1, 33)):
        script_module = torch.jit.script(model)
        base_model_ir = convert_to_graph(script_module, model,
                                         converter=GraphConverterWithShape(), dummy_input=torch.randn(*dummy_input))
        latency = self.latency_predictor.predict(base_model_ir, model_type = 'nni-ir')

        return latency



class HardwareEnergyEstimator:
    def __init__(self, applied_hardware):
        import nn_meter  # pylint: disable=import-error
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
        self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)

    def estimate(self, model, dummy_input=(1, 1, 33)):
        script_module = torch.jit.script(model)
        base_model_ir = convert_to_graph(script_module, model,
                                         converter=GraphConverterWithShape(), dummy_input=torch.randn(*dummy_input))
        latency = self.latency_predictor.predict(base_model_ir, model_type = 'nni-ir')

        return latency
    

class TestNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mutable_all_flatten = nni.retiarii.nn.pytorch.Flatten()
        self.mutable_all_fc_0 = nni.retiarii.nn.pytorch.Linear(in_features=33, out_features=512)
        # self.mutable_all_fc_activation_0 = nni.retiarii.nn.pytorch.Identity()
        self.mutable_all_fc_1 = nni.retiarii.nn.pytorch.Linear(in_features=512, out_features=512)
        # self.mutable_all_fc_activation_1 = nni.retiarii.nn.pytorch.Identity()
        self.mutable_all_fc_2 = nni.retiarii.nn.pytorch.Linear(in_features=512, out_features=512)
        # self.mutable_all_fc_activation_2 = nni.retiarii.nn.pytorch.Identity()
        self.mutable_all_fc_17 = nni.retiarii.nn.pytorch.Linear(in_features=512, out_features=1)

    def forward(self, input__1):
        a = self.mutable_all_flatten(input__1)
        mutable_all_fc_0 = self.mutable_all_fc_0(a)
        # mutable_all_fc_activation_0 = self.mutable_all_fc_activation_0(mutable_all_fc_0)
        mutable_all_fc_1 = self.mutable_all_fc_1(mutable_all_fc_0)
        # mutable_all_fc_activation_1 = self.mutable_all_fc_activation_1(mutable_all_fc_1)
        mutable_all_fc_2 = self.mutable_all_fc_2(mutable_all_fc_1)
        # mutable_all_fc_activation_2 = self.mutable_all_fc_activation_2(mutable_all_fc_2)
        mutable_all_fc_17 = self.mutable_all_fc_17(mutable_all_fc_2)
        return mutable_all_fc_17
    
if __name__ == "__main__":
    model = TestNeuralNetwork()
    # dummy_input = torch.randn((1, 1, 33))
    latency_estimator = HardwareLatencyEstimator('myriadvpu_openvino2019r2')
    latency = latency_estimator.estimate(model)
    print(latency)