# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .utils import get_kernel_name
from .extract_feature import get_predict_features


def merge_conv_kernels(kernelname):
    """
    to speed up, we merge conv and dwconv related kernels into one kernel by their name
    """
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname


def predict_model_latency(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    
    result = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        kernelname = get_kernel_name(rkernel)
        if kernelname in ["add"]:
            continue
        if kernelname == "concat":
            if len(features) > 5:
                features = features[:5]
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict([features]) # in unit of ms
            if "conv" in kernelname:
                result[kernel + '#' + str(layer)] = pys[0]
            else:
                result[kernel + '#' + str(layer)] = pys[0]
    return result


def predict_model_power(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    result = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        kernelname = get_kernel_name(rkernel)
        # if kernelname in ["add"]: # , "concat"
        #     continue
        if kernelname == "concat":
            if len(features) > 5:
                features = features[:5]
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict([features]) # in unit of ms
            result[kernel + '#' + str(layer)] = pys[0]
        
    return result

def nn_predict(predictors, kernel_units):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """
    features = get_predict_features(kernel_units)
    
    if "power" in predictors and "latency" in predictors:
        latency_dict = predict_model_latency(features, predictors["latency"])
        power_dict = predict_model_power(features, predictors["power"])
        energy = 0
        for key in latency_dict:
            if key in power_dict:
                energy += latency_dict[key] * power_dict[key]
        return {"latency": sum(latency_dict.values()), "energy": energy/1000}
    
    elif "latency" in predictors and "power" not in predictors:
        latency_dict = predict_model_power(features, predictors["latency"])
        print(latency_dict)
        return {"latency": sum(latency_dict.values())} 
