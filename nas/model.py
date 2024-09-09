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
            conv_n_layer_options=[1, 2, 3],
            conv_n_filter_options=range(8, 1025, 8),
            conv_op_type='__torch__.nni.retiarii.nn.pytorch.Conv1d',
            in_ch=n_features,
            conv_activation_fn_options=get_activation_name_options(),
            pooling_op_type_options=["__torch__.nni.retiarii.nn.pytorch.MaxPool1d", "__torch__.nni.retiarii.nn.pytorch.AvgPool1d"],
            pooling_kernel_size_options=[3, 5, 7],
            flatten_options=[True, False],
            n_neurons_options=range(8, 1025, 8),
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

@nni.retiarii.model_wrapper
class MLPSpace(nn.Module):
    def __init__(self, n_features=33): # , 
        super().__init__()
        layers = []
        self.n_features = n_features
        
        _layers = nn.Placeholder(
            label='mutable_all',
            n_input_features=n_features,
            n_neurons_option=range(4, 4097, 4),
            fc_n_layer_option=range(2, 10),
            dropout_option=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
            fc_activation_fn_option=get_activation_name_options()
        )
        layers.append(_layers)
        self.layers = nn.Sequential(*layers)
                
    def forward(self, x):
        return self.layers(x)
    
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

@nni.retiarii.model_wrapper
class ResNetSpace(nn.Module):
    class ResNetBlock(nn.Module):
        def __init__(self, idx, d_in=33):
            super().__init__()
            self.batchnorm_1 = nn.LayerChoice([
                    nn.BatchNorm1d(d_in),
                    nn.Identity()
            ], key="resnetblock_isbn_"+str(idx))   
            hidden_dim = nn.ValueChoice(range(4, 513, 4), label="resnetblock_" + str(idx) + "_d_hidden")
            self.linear_1 = nn.Linear(d_in, hidden_dim)
            self.relu_1 = nn.ReLU()
            self.dropout_1 = nn.Dropout(p=nn.ValueChoice([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], label="resnetblock_" + str(idx) + "_droprate_0"))
            self.linear_2 = nn.Linear(hidden_dim, d_in)
            self.dropout_2 = nn.Dropout(p=nn.ValueChoice([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], label="resnetblock_" + str(idx) + "_droprate_1"))

        def forward(self, x):
            x_input = x.squeeze(1)
            x = self.batchnorm_1(x.squeeze(dim=1)) 
            x = self.linear_1(x)
            x = self.relu_1(x)
            x = self.dropout_1(x) 
            x = self.linear_2(x)
            x = self.dropout_2(x)
            return x_input + x 

    class ResNetHead(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.batchnorm = nn.LayerChoice([
                    nn.BatchNorm1d(d_in),
                    nn.Identity()
            ], key="resnethead_isbn")   
            self.relu = nn.ReLU()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, x):
            x = self.batchnorm(x.squeeze(dim=1))
            x = self.relu(x)
            x = self.linear(x)
            return x
            
    def __init__(self, n_features=33): # , 
        super().__init__()
        d_main = nn.ValueChoice(range(4, 257, 4), label="resnet_d_main")
        self.first_layer = nn.Linear(n_features, d_main)
        self.resnetblocks = nn.Repeat(lambda idx: ResNetSpace.ResNetBlock(idx, d_main), (1, 10), label="n_resnetblocks")
        self.head = ResNetSpace.ResNetHead(d_main, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.resnetblocks (x)
        x = self.head(x)
        return x
    
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# @nni.retiarii.model_wrapper
# class ResNetSpace(nn.Module):
#     class ResNetBlock(nn.Module):
#         def __init__(self, idx, d_in=33):
#             super().__init__()
#             _layers = nn.Placeholder(
#                 label='mutable_resnetblock_{idx}',
#                 useBN=[True, False],
#                 dropout_1_options=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
#                 dropout_2_options=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
#                 d_hidden_options=nn.ValueChoice(range(4, 5013, 4))
#             )
#             layers.append(_layers)  
#             self.layers = nn.Sequential(*layers)
                
#         def forward(self, x):
#             return self.layers(x)

#         def forward(self, x):
#             x_input = x.squeeze(1)
#             x = self.batchnorm_1(x.squeeze(dim=1)) 
#             x = self.linear_1(x)
#             x = self.relu_1(x)
#             x = self.dropout_1(x) 
#             x = self.linear_2(x)
#             x = self.dropout_2(x)
#             return x_input + x 

#     class ResNetHead(nn.Module):
#         def __init__(self, d_in, d_out):
#             super().__init__()

#             _layers = nn.Placeholder(
#                 label='mutable_resnethead',
#                 useBN=[True, False]
#             )
#             layers.append(_layers)  
#             self.layers = nn.Sequential(*layers)

#         def forward(self, x):
#             # x = self.batchnorm(x.squeeze(dim=1))
#             # x = self.relu(x)
#             # x = self.linear(x)
#             return self.layers(x)
            
#     def __init__(self, n_features=33): # , 
#         super().__init__()
#         d_main = nn.ValueChoice(range(4, 257, 4))
#         _layers = nn.Placeholder(
#                 label='mutable_resnet',
#                 n_blocks=range(1, 10),
#                 d_main_options=range(4, 257,4)
#             )
#         layers.append(_layers)  
#         self.layers = nn.Sequential(*layers)
#         # self.first_layer = nn.Linear(n_features, d_main)
#         # self.resnetblocks = nn.Repeat(lambda idx: ResNetSpace.ResNetBlock(idx, d_main), (1, 10))
#         # self.head = ResNetSpace.ResNetHead(d_main, 1)

#     def forward(self, x):
#         return self.layers(x)
    
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()