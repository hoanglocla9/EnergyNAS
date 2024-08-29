from model import *
from data import *

from nn_meter import load_latency_predictor
import os
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))
data_dir = "./data"
# Load sensor raw data into a pandas data frame
rawdf = pd.read_csv(os.path.join(data_dir,'MISO_220621-0627_1mnDataset_K96.csv'))

# class _model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.__layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(33, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#     def __init__(self):
#         super().__init__()
#         self.__layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(33, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x__1):
#         __layers = self.__layers(x__1)
#         return __layers
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x__1):
#         __layers = self.__layers(x__1)
#         return __layers

class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x__1):
        __layers = self.__layers(x__1)
        return __layers


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

@nni.trace
class MyDataset(Dataset):
    def __init__(self, lag_range, target):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_data(lag_range, target)
        self.x=torch.tensor(self.preprocessed_df[self.features].values,dtype=torch.float32).reshape(-1, 1, len(self.features))
        self.y=torch.tensor(self.preprocessed_df[self.target].values,dtype=torch.float32).reshape(-1, 1).to(device)
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x).to(device)
        
    def load_data(self, lag_range, target):
        #parent_dir = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))
        data_dir = "./data/" #os.path.join(parent_dir,"data")
        # Load sensor raw data into a pandas data frame
        rawdf = pd.read_csv(os.path.join(data_dir,'MISO_220621-0627_1mnDataset_K96.csv'))
        
        
        self.features = ['pressure(hPa)', 'rh(%)', 'temp_sensor(C)', 'k96_lpl_raw', 'k96_spl_raw', 'k96_mpl_raw', "k96_h2o(ppm) ", "k96_ch4(ppm) "]
        if target == "ref_h2o(ppm)":
            self.features += ["k96_h2o(ppm) _" + str(i) for i in range(1, lag_range)]
            extra_prefix = "k96_h2o(ppm) " 
        elif target == "ref_ch4(ppm)":
            self.features += ["k96_ch4(ppm) _" + str(i) for i in range(1, lag_range)]
            extra_prefix = "k96_ch4(ppm) " 
            
        values = nni.trace(DataFrame)(rawdf[extra_prefix].values)
        lag_df = concat([values.shift(j) for j in range(lag_range)], axis=1)
        lag_df.columns = [extra_prefix + "_" +str(j) for j in range(lag_range)]
        self.preprocessed_df =  pd.concat([rawdf, lag_df], axis=1)[lag_range:]
        self.target = target
        
        
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    


class HardwareLatencyEstimator:
    def __init__(self, applied_hardware):
        import nn_meter  # pylint: disable=import-error
        self.predictor_name = applied_hardware
        self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)

    def estimate(self, model, dummy_input=(1, 1, 33)):
        model = adjust_model_code(model)
        script_module = torch.jit.script(model)
        from nni.retiarii.converter import convert_to_graph
        from nni.retiarii.converter.graph_gen import GraphConverterWithShape
        base_model_ir = convert_to_graph(script_module, model,
                                         converter=GraphConverterWithShape(), dummy_input=torch.randn(*dummy_input))
        latency = self.latency_predictor.predict(base_model_ir, model_type = 'nni-ir')

        return latency

def my_import(name):
    print(name)
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def reverse_model_code(path):
    model_code = ""
    with open (path, "r") as f:
        model_code = f.read()

    pattern = """class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers"""
    
    if pattern not in model_code:
        model_code += pattern

    model_code = model_code.replace("import nni.nas.nn.pytorch", "")

    with open (path, "w") as f:
        f.write(model_code)

    
def adjust_model_code(model):
    import inspect, re
    path = inspect.getfile(model.__class__)
    model_code = ""
    with open (path, "r") as f:
        model_code = f.read()

    model_code += """class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self._layers = _model__layers()
        self._mapping_ = {'_layers': 'layers'}

    def forward(self, x__1):
        _layers = self._layers(x__1)
        return _layers"""



    model_code = re.sub(r"class _model\(nn\.Module\)(.*\n)*.*return __layers", "", model_code)
    with open (path, "w") as f:
        f.write(model_code)
            
    module_name = ".".join(path.split("/")[-3:])[:-3] + "._model__layers"
        
    _model__layers = my_import(module_name)
    return _model__layers()

def latency_reward_component(latency, target_latency):
    alpha, beta = -0.07, -0.07 # as in the paper  https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf
    if latency < target_latency:
        return (latency * 1.0 /target_latency * 1.0) ** alpha
    else:
        return (latency * 1.0 /target_latency * 1.0) ** beta

def reward_function(accuracy, latency, target_latency):
    latency_component = latency_reward_component(latency, target_latency)
    normalized_accuracy = (accuracy - 0.35) / ( 8-0.35)
    return normalized_accuracy * latency_component

def evaluate_model(model_cls, metric, target):
    final_metrics = {"default": [], "MSE": [], "MAE": [], "Latency": []}
    optimized_metric = "MAE"
    target = "ref_{}(ppm)".format(target)
    lag_range = 26
    if "Latency" in metric:
        latency_estimator = nni.trace(HardwareLatencyEstimator)('myriadvpu_openvino2019r2')
    else:
        latency_estimator = None

    if latency_estimator != None:
        latency = latency_estimator.estimate(model_cls())
    else:
        latency = None

    k_folds = 3
    kfold = KFold(n_splits=k_folds, shuffle=True)
    dataset = nni.trace(MyDataset)(lag_range, target)
    criterion = torch.nn.L1Loss()
    acc_fn = torch.nn.MSELoss()

    average_loss = 0.0
    average_min = 0.0

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_ids, torch.Generator().manual_seed(42))
        valid_subsampler = SubsetRandomSampler(valid_ids, torch.Generator().manual_seed(42))

        train_loader = nni.trace(DataLoader)(dataset, batch_size=512, sampler=train_subsampler)
        valid_loader = nni.trace(DataLoader)(dataset, batch_size=512, sampler=valid_subsampler)
        model = model_cls()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device) 

        # dummy_input = torch.zeros(1, 1, 33).to(device)
        # torch.onnx.export(model, (dummy_input, ), os.path.join(os.environ['NNI_OUTPUT_DIR'], 'model.onnx'))

        model.apply(reset_weights)
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
                    valid_accuracy += acc_fn(output, tensor_y).item() * len(tensor_x)


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
                train_accuracy += acc_fn(output, tensor_y).item() * len(tensor_x)

            train_loss = train_loss / len(train_loader.sampler.indices)
            train_accuracy = train_accuracy / len(train_loader.sampler.indices)
            
            if optimized_metric == "MAE":
                intermediate_metrics = {"default": -1.0 * valid_loss, "MSE": -1.0 * valid_accuracy}
            else:
                intermediate_metrics = {"default": -1.0 * valid_accuracy, "MAE": -1.0 * valid_loss}
            
        
        if latency != None:
            final_metrics["default"].append(reward_function(min(min_loss_list), latency, target_latency=5))
        else:
            final_metrics["default"].append(-1.0 * min(min_loss_list))
        final_metrics["MSE"].append(-1.0 * min(min_acc_list))
        final_metrics["MAE"].append(-1.0 * min(min_loss_list))
        
    # print('Average Loss: {} Average Accuracy: {}'.format(np.mean(final_metrics["default"]), np.mean(final_metrics["MSE"])))
    # print('STD Loss: {} STD Accuracy: {}'.format(np.std(final_metrics["default"]), np.std(final_metrics["MSE"])))
    # torch.onnx.export(model, input, "test.onnx", input_names=['input'],
    #               output_names=['output'], export_params=True)
    dummy_input = torch.zeros(1, 1, 33).to(device)
    torch.onnx.export(model, (dummy_input, ), "test.onnx", export_params=True)
    
    return np.mean(final_metrics["MAE"]), np.std(final_metrics["MAE"]), np.mean(final_metrics["MSE"]), np.std(final_metrics["MSE"]), latency, np.mean(final_metrics["default"])


# metrics =  [ "MAExLatency"] # "",
# algorithms = ["evolution", "reinforce"] #, "evolution"
# trial_opts = [200]
# targets = ["h2o"] # , "h2o"
# top_k = 10


# for n_trials in trial_opts:
#     for target in targets:
#         for metric in metrics:
#             for algo in algorithms:
#                 output_str = ""
#                 output_path = "outputs/{}_{}_{}_{}.txt".format(algo, target, metric, n_trials)
#                 for i in range(1, top_k +1):
#                     print ("[+] Evaluate {}_{}_{}_{}_model top {}: ".format(algo, target, metric, n_trials, i))
#                     _model = None
#                     module_name = "results.{}_{}_{}_{}.top_{}".format(algo, target, metric, n_trials, i)
#                     import_str = "from {} import _model".format(module_name)
#                     try:
#                         exec(import_str)
#                     except Exception as e:
#                         print("Retry to import {} ---".format(import_str))
#                         print(e)
#                         try:
#                             reverse_model_code(module_name.replace(".", "/")+".py")
#                             exec(import_str)
#                         except Exception as e:
#                             print("Cannot import {}!!!".format(import_str)) 
#                             print(e)
#                             continue

#                     loss_mean, loss_std, accuracy_mean, accuracy_std, latency, score = evaluate_model(_model, metric, target)
#                     output_str += "Top {} Model --- Loss: {} +/- {},  Accuracy: {} +/- {},  Latency: {}, Score: {}\n".format(i, loss_mean, loss_std, accuracy_mean, accuracy_std, latency, score)

#                 if output_str != "":
#                     with open(output_path, "w") as f:
#                         f.write(output_str)

print(evaluate_model(_model, "MAE", "h2o"))