import nni
import nni.retiarii.nn.pytorch as nn

    
def get_activation_name_options():
    return [
        # "__torch__.nni.retiarii.nn.pytorch.Identity", 
        "None", ## replace for Identity
        "__torch__.nni.retiarii.nn.pytorch.ReLU", 
        "__torch__.nni.retiarii.nn.pytorch.Tanh", 
        "__torch__.nni.retiarii.nn.pytorch.LeakyReLU", 
        "__torch__.nni.retiarii.nn.pytorch.Hardshrink", 
        "__torch__.nni.retiarii.nn.pytorch.Hardsigmoid", 
        "__torch__.nni.retiarii.nn.pytorch.Hardtanh", 
        "__torch__.nni.retiarii.nn.pytorch.Hardswish", 
        "__torch__.nni.retiarii.nn.pytorch.LogSigmoid", 
        "__torch__.nni.retiarii.nn.pytorch.PReLU", 
        "__torch__.nni.retiarii.nn.pytorch.ELU", 
        "__torch__.nni.retiarii.nn.pytorch.ReLU6", 
        "__torch__.nni.retiarii.nn.pytorch.RReLU", 
        "__torch__.nni.retiarii.nn.pytorch.CELU",
        "__torch__.nni.retiarii.nn.pytorch.SiLU",
        # "__torch__.nni.retiarii.nn.pytorch.Mish",
        "__torch__.nni.retiarii.nn.pytorch.Softplus",
        "__torch__.nni.retiarii.nn.pytorch.Softshrink",
        "__torch__.nni.retiarii.nn.pytorch.Softsign",
        "__torch__.nni.retiarii.nn.pytorch.Tanhshrink" 
    ] 
@nni.retiarii.model_wrapper
class CalibrationModelSpace(nn.Module):
    def __init__(self, n_features=33): # , 
        super().__init__()
        layers = []
        self.n_features = n_features
        
        _layers = nn.Placeholder(
            label='mutable_all',
            conv_kernel_size_options=[1, 3, 5, 7],
            conv_n_layer_options=[1, 2],
            conv_n_filter_options=[32, 64, 128, 256, 512],
            conv_op_type='__torch__.nni.retiarii.nn.pytorch.Conv1d',
            in_ch=n_features,
            conv_activation_fn_options=get_activation_name_options(),
            pooling_op_type_options=["__torch__.nni.retiarii.nn.pytorch.MaxPool1d", "__torch__.nni.retiarii.nn.pytorch.AvgPool1d"],
            pooling_kernel_size_options=[3, 5, 7],
            flatten_options=[True, False],
            n_neurons_options=[16, 32, 64, 128, 256, 512],
            fc_n_layer_options=list(range(1, 19)),
            fc_op_type='__torch__.nni.retiarii.nn.pytorch.Linear',
            fc_activation_fn_options=get_activation_name_options()
        )
        layers.append(_layers)  
        self.layers = nn.Sequential(*layers)
                
    def forward(self, x):
        return self.layers(x)
    
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()