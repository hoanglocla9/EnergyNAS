import nni
import os

import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from .data import MISO_Data_v1
from sklearn.model_selection import KFold
from .estimator import HardwareMetricEstimator
from torchmetrics.regression import MeanAbsolutePercentageError

from nni.nas.evaluator.pytorch import Lightning, Trainer, DataLoader
from .darts import DartsRegressionModule
import copy
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
def evaluate_model_darts(lag_range, target, n_gpus=0, max_epochs=50, fast_dev_run=False):
    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    train_set, valid_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
    search_train_loader, search_valid_loader = nni.trace(
        DataLoader)(train_set),  nni.trace(DataLoader)(valid_set)

    evaluator = Lightning(
        DartsRegressionModule(0.025, 3e-4, 0., max_epochs),
        Trainer(
            max_epochs=max_epochs,
            fast_dev_run=fast_dev_run,
        ),
        train_dataloaders=search_train_loader,
        val_dataloaders=search_valid_loader
    )
    return evaluator


@nni.trace
def evaluate_model(model_cls, lag_range, target, performance_metric, efficiency_metric, target_values, mode="filter", target_device="myriadvpu_openvino2019r2"):
    """
        target_device: the target device name. We support two platforms, namely myriadvpu_openvino2019r2, jetsonnano_jetpack46.
        mode: mode for NAS problem, we support filter and multi-objective mode. Filter mode is suitable for contraint single objective problem.
        optimized_metrics: Optimized metrics, we support following metrics: MSE, MPAE, latency and energy. ("energy" option also includes "latency")
        target: name of the targeted feature.
        lag_range: number of time lags.
    """
    final_metrics = {"default": [], "MPAE": [], "MSE": []}
    hardware_metrics = []

    torch.manual_seed(0)
    if mode in ["debug"]:
        estimator = None
    else:
        final_metrics[efficiency_metric] = []
        hardware_metrics = [efficiency_metric]
        if efficiency_metric == "energy":  # add latency
            hardware_metrics.append("latency")
            final_metrics["latency"] = []
        estimator = nni.trace(HardwareMetricEstimator)(
            target_device, hardware_metrics)
        hardware_estimated_result = estimator.estimate(model_cls())

    k_folds = 3
    kfold = nni.trace(KFold)(n_splits=k_folds, shuffle=True)
    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    criterion = torch.nn.MSELoss()
    acc_fn = MeanAbsolutePercentageError()  # torch.nn.MSELoss()

    average_loss = 0.0
    average_min = 0.0

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = nni.trace(SubsetRandomSampler)(
            train_ids, torch.Generator().manual_seed(42))
        valid_subsampler = nni.trace(SubsetRandomSampler)(
            valid_ids, torch.Generator().manual_seed(42))

        train_loader = nni.trace(DataLoader)(
            dataset, batch_size=512, sampler=train_subsampler)
        valid_loader = nni.trace(DataLoader)(
            dataset, batch_size=512, sampler=valid_subsampler)
        model = model_cls()
        dummy_input = torch.zeros(1, 1, 33)
        torch.onnx.export(model, (dummy_input, ), os.path.join(
            os.environ['NNI_OUTPUT_DIR'], 'model.onnx'))

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        min_loss_list = []
        min_acc_list = []
        for epoch in range(100):
            with torch.no_grad():
                valid_loss = 0
                valid_accuracy = 0
                for tensor_x, tensor_y in valid_loader:
                    tensor_x = tensor_x.float()
                    tensor_y = tensor_y.float().reshape(-1, 1)
                    output = model(tensor_x)
                    loss = criterion(output, tensor_y)
                    valid_loss += loss.item() * len(tensor_x)
                    valid_accuracy += acc_fn(output.squeeze().cpu(),
                                             tensor_y.squeeze().cpu()).item() * len(tensor_x)

                valid_loss = valid_loss / len(valid_loader.sampler.indices)
                valid_accuracy = valid_accuracy / \
                    len(valid_loader.sampler.indices)
                min_loss_list.append(valid_loss)
                min_acc_list.append(valid_accuracy)

            train_loss = 0
            train_accuracy = 0
            for tensor_x, tensor_y in train_loader:
                tensor_x = tensor_x.float()
                tensor_y = tensor_y.float().reshape(-1, 1)
                optimizer.zero_grad()
                output = model(tensor_x)
                loss = criterion(output, tensor_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(tensor_x)
                train_accuracy += acc_fn(output.squeeze().cpu(),
                                         tensor_y.squeeze().cpu()).item() * len(tensor_x)

            train_loss = train_loss / len(train_loader.sampler.indices)
            train_accuracy = train_accuracy / len(train_loader.sampler.indices)

            intermediate_metrics = {"MPAE": -1.0 *
                                    valid_accuracy, "MSE": -1.0 * valid_loss}
            intermediate_metrics["default"] = intermediate_metrics[performance_metric]
            nni.report_intermediate_result(intermediate_metrics)

        final_metrics["MPAE"].append(-1.0 *
                                     sum(min_acc_list)/len(min_acc_list))
        final_metrics["MSE"].append(-1.0 *
                                    sum(min_loss_list)/len(min_loss_list))

        if estimator != None:
            for hw_metric in hardware_metrics:
                final_metrics[hw_metric].append(
                    hardware_estimated_result[hw_metric])
            if mode == "moo_v1":
                final_metrics["default"].append(reward_function_v2(-1.0 * final_metrics[performance_metric][-1],
                                                hardware_estimated_result[efficiency_metric], target_values[performance_metric]))
            elif mode in ["filter", "moo_v2"]:
                final_metrics["default"].append(
                    final_metrics[performance_metric][-1])
            else:
                raise Exception(f"Not Support this mode \"{mode}\" yet!")
        else:
            final_metrics["default"].append(
                final_metrics[performance_metric][-1])

    metric = {}
    for key in final_metrics:
        if len(final_metrics[key]) == 0:
            metric[key] = 0
        else:
            metric[key] = np.mean(final_metrics[key])
    if mode == "moo_v2":
        metric['moo'] = {"obj_1": metric[performance_metric],
                         'obj_2': metric[efficiency_metric]}
        # report final test result
    print("Final result: " + str(metric))
    nni.report_final_result(metric)


@nni.trace
def evaluate_model_v2(model_cls, batch_size, lag_range, target, performance_metric, efficiency_metric, target_values, mode="filter", target_device="myriadvpu_openvino2019r2"):
    """
        target_device: the target device name. We support two platforms, namely myriadvpu_openvino2019r2, jetsonnano_jetpack46.
        mode: mode for NAS problem, we support filter and multi-objective mode. Filter mode is suitable for contraint single objective problem.
        optimized_metrics: Optimized metrics, we support following metrics: MSE, MPAE, latency and energy. ("energy" option also includes "latency")
        target: name of the targeted feature.
        lag_range: number of time lags.
    """
    final_metrics = {"default": [], "MPAE": [], "MSE": []}
    hardware_metrics = []

    if mode in ["debug"]:
        estimator = None
    else:
        final_metrics[efficiency_metric] = []
        hardware_metrics = [efficiency_metric]
        if efficiency_metric == "energy":  # add latency
            hardware_metrics.append("latency")
            final_metrics["latency"] = []
        estimator = nni.trace(HardwareMetricEstimator)(
            target_device, hardware_metrics)
        hardware_estimated_result = estimator.estimate(model_cls())

    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    criterion = torch.nn.MSELoss()
    acc_fn = MeanAbsolutePercentageError()  # torch.nn.MSELoss()

    torch.manual_seed(0)
    average_loss = 0.0
    average_min = 0.0

    train_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

    train_loader = nni.trace(DataLoader)(
        train_set, batch_size=batch_size, shuffle=True)
    valid_loader = nni.trace(DataLoader)(
        val_set, batch_size=batch_size, shuffle=True)
    model = model_cls()
    dummy_input = torch.zeros(1, 33)  # (1,1,33)
    torch.onnx.export(model, (dummy_input, ), os.path.join(
        os.environ['NNI_OUTPUT_DIR'], 'model.onnx'))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(device)
    # model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    min_loss_list = []
    min_acc_list = []
    best_valid_accuracy = np.inf
    for epoch in range(500):
        train_loss = 0
        train_accuracy = 0
        for tensor_x, tensor_y in train_loader:
            print(tensor_x.shape)
            tensor_x = tensor_x.float()
            tensor_y = tensor_y.float().reshape(-1, 1)
            optimizer.zero_grad()
            output = model(tensor_x)
            loss = criterion(output, tensor_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(tensor_x)
            train_accuracy += acc_fn(output.squeeze().cpu(),
                                     tensor_y.squeeze().cpu()).item() * len(tensor_x)

        train_loss = train_loss / len(train_loader.sampler.indices)
        train_accuracy = train_accuracy / len(train_loader.sampler.indices)

        with torch.no_grad():
            valid_loss = 0
            valid_accuracy = 0
            for tensor_x, tensor_y in valid_loader:
                tensor_x = tensor_x.float()
                tensor_y = tensor_y.float().reshape(-1, 1)
                output = model(tensor_x)
                loss = criterion(output, tensor_y)
                valid_loss += loss.item() * len(tensor_x)
                valid_accuracy += acc_fn(output.squeeze().cpu(),
                                         tensor_y.squeeze().cpu()).item() * len(tensor_x)

            valid_loss = valid_loss / len(valid_loader.sampler.indices)
            valid_accuracy = valid_accuracy / \
                len(valid_loader.sampler.indices)
            min_loss_list.append(valid_loss)
            min_acc_list.append(valid_accuracy)

        if valid_accuracy < best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            counter = 0
        else:
            counter += 1

        if counter > 50:
            break

        intermediate_metrics = {"MPAE": -1.0 *
                                valid_accuracy, "MSE": -1.0 * valid_loss}
        intermediate_metrics["default"] = intermediate_metrics[performance_metric]
        nni.report_intermediate_result(intermediate_metrics)

        final_metrics["MPAE"].append(-1.0 *
                                     sum(min_acc_list)/len(min_acc_list))
        final_metrics["MSE"].append(-1.0 *
                                    sum(min_loss_list)/len(min_loss_list))

        if estimator != None:
            for hw_metric in hardware_metrics:
                final_metrics[hw_metric].append(
                    hardware_estimated_result[hw_metric])
            if mode == "moo_v1":
                final_metrics["default"].append(reward_function_v2(-1.0 * final_metrics[performance_metric][-1],
                                                hardware_estimated_result[efficiency_metric], target_values[performance_metric]))
            elif mode in ["filter", "moo_v2"]:
                final_metrics["default"].append(
                    final_metrics[performance_metric][-1])
            else:
                raise Exception(f"Not Support this mode \"{mode}\" yet!")
        else:
            final_metrics["default"].append(
                final_metrics[performance_metric][-1])

    metric = {}
    for key in final_metrics:
        if len(final_metrics[key]) == 0:
            metric[key] = 0
        else:
            metric[key] = np.mean(final_metrics[key])
    if mode == "moo_v2":
        metric['moo'] = {"obj_1": metric[performance_metric],
                         'obj_2': metric[efficiency_metric]}
        # report final test result
    print("Final result: " + str(metric))
    nni.report_final_result(metric)
