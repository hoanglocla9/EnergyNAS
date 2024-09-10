import nni, os

import numpy as np 

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from .data import MISO_Data_v1
from sklearn.model_selection import KFold
from .estimator import  HardwareMetricEstimator
from torchmetrics.regression import MeanAbsolutePercentageError

@nni.trace
def latency_reward_component(latency, target_latency):
    alpha, beta = -0.07, -0.07 # as in the paper  https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf
    if latency < target_latency:
        return (latency * 1.0 /target_latency * 1.0) ** alpha
    else:
        return (latency * 1.0 /target_latency * 1.0) ** beta
@nni.trace
def reward_function(accuracy, latency, target_latency):
    latency_component = latency_reward_component(latency, target_latency)
    # normalized_accuracy = (accuracy - 0.35) / ( 8-0.35)
    # accuracy
    return -1.0 * accuracy * latency_component

@nni.trace
def evaluate_model(model_cls, lag_range, target, optimized_metrics, target_values, mode="filter", target_device="myriadvpu_openvino2019r2"):
    """
        target_device: the target device name. We support two platforms, namely myriadvpu_openvino2019r2, jetsonnano_jetpack46.
        mode: mode for NAS problem, we support filter and multi-objective mode. Filter mode is suitable for contraint single objective problem.
        optimized_metrics: Optimized metrics, we support following metrics: MAE, MPAE, latency and energy. ("energy" option also includes "latency")
        target: name of the targeted feature.
        lag_range: number of time lags.
    """
    final_metrics = {"default": [], "MPAE": [], "MAE": []}
    hardware_metrics = []
    if "energy" in optimized_metrics:
        final_metrics[ "latency"] = []
        final_metrics["energy"] = []
        hardware_metrics = ["energy", "latency"]
        estimator = nni.trace(HardwareMetricEstimator)(target_device, hardware_metrics)
    elif "latency" in optimized_metrics:
        final_metrics[ "latency"] = []
        hardware_metrics = ["latency"]
        estimator = nni.trace(HardwareMetricEstimator)(target_device, hardware_metrics)
    else:
        estimator = None
    
    if estimator != None:
        hardware_estimated_result = estimator.estimate(model_cls())

    k_folds = 3
    kfold = nni.trace(KFold)(n_splits=k_folds, shuffle=True)
    dataset = nni.trace(MISO_Data_v1)(lag_range, target)
    criterion = nni.trace(torch.nn.L1Loss())
    acc_fn = nni.trace(MeanAbsolutePercentageError())## torch.nn.MSELoss()

    average_loss = 0.0
    average_min = 0.0

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = nni.trace(SubsetRandomSampler)(train_ids, torch.Generator().manual_seed(42))
        valid_subsampler = nni.trace(SubsetRandomSampler)(valid_ids, torch.Generator().manual_seed(42))

        train_loader = nni.trace(DataLoader)(dataset, batch_size=512, sampler=train_subsampler)
        valid_loader = nni.trace(DataLoader)(dataset, batch_size=512, sampler=valid_subsampler)
        model = model_cls()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device) 

        dummy_input = torch.zeros(1, 1, 33).to(device)
        torch.onnx.export(model, (dummy_input, ), os.path.join(os.environ['NNI_OUTPUT_DIR'], 'model.onnx'))

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
                    valid_accuracy += acc_fn(output.squeeze().cpu(), tensor_y.squeeze().cpu()).item() * len(tensor_x)


                valid_loss = valid_loss / len(valid_loader.sampler.indices)
                valid_accuracy = valid_accuracy / len(valid_loader.sampler.indices)
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
                train_accuracy += acc_fn(output.squeeze().cpu(), tensor_y.squeeze().cpu()).item() * len(tensor_x)

            train_loss = train_loss / len(train_loader.sampler.indices)
            train_accuracy = train_accuracy / len(train_loader.sampler.indices)
            
            if "MAE" in optimized_metrics:
                intermediate_metrics = {"default": -1.0 * valid_loss, "MPAE": -1.0 * valid_accuracy, "MAE": -1.0 * valid_loss}
            else:
                intermediate_metrics = {"default": -1.0 * valid_accuracy, "MPAE": -1.0 * valid_accuracy, "MAE": -1.0 * valid_loss}
            nni.report_intermediate_result(intermediate_metrics)
        
        if estimator != None:
            # hardware_estimated_result = estimator.estimate(model_cls())
            for hw_metric in hardware_metrics:
                final_metrics[hw_metric].append(hardware_estimated_result[hw_metric])

            final_metrics["MPAE"].append(-1.0 * sum(min_acc_list)/len(min_acc_list))
            final_metrics["MAE"].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
            if mode == "mmo":
                if "MAE" in optimized_metrics:
                    final_metrics["default"].append(reward_function(sum(min_loss_list)/len(min_loss_list), hardware_estimated_result['latency'], target_values["latency"]))
                else:
                    final_metrics["default"].append(reward_function(sum(min_acc_list)/len(min_acc_list), hardware_estimated_result['latency'], target_values["latency"]))
            elif mode == "filter":
                if "MAE" in optimized_metrics:
                    final_metrics["default"].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
                else:
                    final_metrics["default"].append(-1.0 * sum(min_acc_list)/len(min_acc_list))
            elif mode == "debug":
                if "MAE" in optimized_metrics:
                    final_metrics["default"].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
                else:
                    final_metrics["default"].append(-1.0 * sum(min_acc_list)/len(min_acc_list))
            else:
                raise Exception (f"Not Support this mode \"{mode}\" yet!")
        else:
            final_metrics["MPAE"].append(-1.0 * sum(min_acc_list)/len(min_acc_list))
            final_metrics["MAE"].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
            if optimized_metrics == "MAE":
                final_metrics["default"].append(-1.0 * sum(min_loss_list)/len(min_loss_list))
            else:
                final_metrics["default"].append(-1.0 * sum(min_acc_list)/len(min_acc_list))
    metric = {}
    for key in final_metrics:
        if len(final_metrics[key]) == 0:
            metric[key] = 0
        else:
            metric[key] = np.mean(final_metrics[key])

    # report final test result
    print("Final result: " + str(metric))
    nni.report_final_result(metric)