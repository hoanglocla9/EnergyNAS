import nni
import os

import numpy as np

from torch.utils.data import DataLoader
import torch
from .data import MISO_Data_v1
from .estimator import HardwareMetricEstimator
from torchmetrics.regression import MeanAbsolutePercentageError

from nni.retiarii.evaluator import FunctionalEvaluator
from nni.nas.evaluator.pytorch import Lightning, Trainer, DataLoader
from .darts import DartsRegressionModule
from nni.nas.evaluator.pytorch.lightning import Regression
import torch


@nni.trace
def latency_reward_component(latency, target_latency):
    alpha, beta = -0.07, -0.07  # as in the paper  https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf
    if latency < target_latency:
        return (latency * 1.0 / target_latency * 1.0) ** alpha
    else:
        return (latency * 1.0 / target_latency * 1.0) ** beta


@nni.trace
def reward_function(accuracy, latency, target_latency):
    latency_component = latency_reward_component(latency, target_latency)
    # normalized_accuracy = (accuracy - 0.35) / ( 8-0.35)
    # accuracy
    return -1.0 * accuracy * latency_component


@nni.trace
def accuracy_reward_component(accuracy, target_accuracy):
    alpha, beta = 2, -1  # -0.07, -0.07 # as in the paper  https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf
    if accuracy < target_accuracy:
        return (accuracy * 1.0 / target_accuracy * 1.0) ** alpha
    else:
        return (accuracy * 1.0 / target_accuracy * 1.0) ** beta


@nni.trace
def reward_function_v2(accuracy, energy, target_accuracy):
    accuracy_component = latency_reward_component(accuracy, target_accuracy)
    # normalized_accuracy = (accuracy - 0.35) / ( 8-0.35)
    # accuracy
    return -1.0 * energy * accuracy_component



@nni.trace
def evaluate_model(model_cls, batch_size, lag_range, target, max_epochs, performance_metric, efficiency_metric, target_values, strategy="random", mode="single", target_device="myriadvpu_openvino2019r2"):
    """
        target_device: the target device name. We support two platforms, namely myriadvpu_openvino2019r2, jetsonnano_jetpack46.
        mode: mode for NAS problem, we support single and multi-objective mode. Filter mode is suitable for contraint single objective problem.
        optimized_metrics: Optimized metrics, we support following metrics: MSE, MPAE, latency and energy. ("energy" option also includes "latency")
        target: name of the targeted feature.
        lag_range: number of time lags.
    """
    final_metrics = {"default": [], performance_metric: []}
    hardware_metrics = []

    model = model_cls()
    if mode == "single":
        estimator = None
    else:
        final_metrics[efficiency_metric] = []
        hardware_metrics = [efficiency_metric]
        if efficiency_metric == "energy":  # add latency
            hardware_metrics.append("latency")
            final_metrics["latency"] = []
        estimator = nni.trace(HardwareMetricEstimator)(
            target_device, hardware_metrics)
        hardware_estimated_result = estimator.estimate(model)

    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    
    if performance_metric == "MSE":
        criterion = torch.nn.MSELoss() 
    else:
        criterion = MeanAbsolutePercentageError() 

    torch.manual_seed(0)

    train_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

    train_loader = nni.trace(DataLoader)(
        train_set, batch_size=batch_size, shuffle=True)
    valid_loader = nni.trace(DataLoader)(
        val_set, batch_size=batch_size, shuffle=True)
    # dummy_input = torch.zeros(1, 33)  # (1,1,33)
    # torch.onnx.export(model, (dummy_input, ), os.path.join(
    #     os.environ['NNI_OUTPUT_DIR'], 'model.onnx'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    min_loss_list = []
    best_valid_loss = np.inf
    for epoch in range(max_epochs):
        train_loss = 0
        for tensor_x, tensor_y in train_loader:
            tensor_x = tensor_x.float()
            tensor_y = tensor_y.float().reshape(-1, 1)
            optimizer.zero_grad()
            output = model(tensor_x)
            loss = criterion(output, tensor_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(tensor_x)
            
        train_loss = train_loss / len(train_loader.sampler)

        with torch.no_grad():
            valid_loss = 0
            for tensor_x, tensor_y in valid_loader:
                tensor_x = tensor_x.float()
                tensor_y = tensor_y.float().reshape(-1, 1)
                output = model(tensor_x)
                loss = criterion(output, tensor_y)
                valid_loss += loss.item() * len(tensor_x)

            valid_loss = valid_loss / len(valid_loader.sampler)
            
            min_loss_list.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1

        if counter > 50:
            break

        intermediate_metrics = {performance_metric: -1.0 * valid_loss}
        intermediate_metrics["default"] = intermediate_metrics[performance_metric]
        nni.report_intermediate_result(intermediate_metrics)
        final_metrics[performance_metric].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
        if mode  == "multi":
            for hw_metric in hardware_metrics:
                final_metrics[hw_metric].append(
                    hardware_estimated_result[hw_metric])
            if strategy in ["reinforce", "random"]:
                final_metrics["default"].append(reward_function_v2(-1.0 * final_metrics[performance_metric][-1],
                                                hardware_estimated_result[efficiency_metric], target_values[performance_metric]))
            elif strategy in ["evolution"]:
                final_metrics["default"].append(final_metrics[performance_metric][-1])
            else:
                raise Exception(f"Not Support mode \"{mode}\" and strategy \"{strategy}\" at the same time yet!")
        else:
            final_metrics["default"].append(final_metrics[performance_metric][-1])

    reported_metrics = {}
    for key in final_metrics:
        if len(final_metrics[key]) == 0:
            reported_metrics[key] = 0
        else:
            reported_metrics[key] = np.mean(final_metrics[key])
    if mode == "multi" and strategy == "evolution":
        reported_metrics['moo'] = {"obj_1": reported_metrics[performance_metric],
                         'obj_2': reported_metrics[efficiency_metric]}
        
    nni.report_final_result(reported_metrics)


@nni.trace
def get_regressor(lag_range, 
                  target, 
                  performance_metric=None,
                  efficiency_metric=None,
                  mode=None,
                  target_values=None,
                  batch_size=32, 
                  max_epochs=50, 
                  learning_rate=0.0001, 
                  weight_decay=1e-5, 
                  fast_dev_run=False, 
                  strategy="random_oneshot"):
    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    train_set, valid_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
    train_loader, valid_loader = nni.trace(
        DataLoader)(train_set, batch_size=batch_size, shuffle=True),  nni.trace(DataLoader)(valid_set, batch_size=batch_size, shuffle=True)
    
    if strategy == "darts":
        evaluator = Lightning(
            DartsRegressionModule(learning_rate, weight_decay, 0., max_epochs),
            Trainer(
                max_epochs=max_epochs,
                fast_dev_run=fast_dev_run,
            ),
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )
    elif strategy == "random_oneshot":
        if performance_metric == "MSE":
            criterion = torch.nn.MSELoss
        else:
            criterion = MeanAbsolutePercentageError
        torch.manual_seed(0)  
        evaluator = Regression(criterion=criterion,
                 learning_rate=learning_rate,
                 weight_decay=weight_decay,
                 optimizer =torch.optim.Adam,
                 train_dataloaders=train_loader,
                 val_dataloaders=valid_loader,
                 max_epochs=max_epochs,
                 export_onnx=True)
    else:
        evaluator = FunctionalEvaluator(evaluate_model,
                                    strategy=strategy,
                                    lag_range=lag_range,
                                    target=target,
                                    performance_metric=performance_metric,
                                    efficiency_metric=efficiency_metric,
                                    mode=mode,
                                    max_epochs=max_epochs,
                                    target_values=target_values, 
                                    batch_size=batch_size)
    return evaluator
