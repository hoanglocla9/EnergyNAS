import nni
import onnx

@nni.trace
def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
@nni.trace
def adjust_model_code(model):
    import inspect, re
    path = inspect.getfile(model.__class__)
    model_code = ""
    with open (path, "r") as f:
        model_code = f.read()

    model_code = re.sub(r"class _model\(nn\.Module\)(.*\n)*.*return __layers", "", model_code)
    model_code = re.sub(r"\*_inputs", "_inputs", model_code)
    model_code = re.sub(r"_inputs\[0\]", "_inputs", model_code)
    # model_code = re.sub(r"import torch", "import nni", model_code)

    model_code = re.sub(r"torch.nn.modules.linear.", "nni.retiarii.nn.pytorch.", model_code)
    model_code = re.sub(r"torch.nn.modules.activation", "nni.retiarii.nn.pytorch.", model_code)
    model_code = re.sub(r"torch.nn.modules.dropout", "nni.retiarii.nn.pytorch.", model_code)
    model_code = re.sub(r"torch.nn.modules.batchnorm", "nni.retiarii.nn.pytorch.", model_code)

    with open (path, "w") as f:
        f.write(model_code)
    try:
        module_name = path.replace('/', '.')[-26:-3] +"._model__layers"
        _model__layers = my_import(module_name)
    except:
        module_name = path.replace('/', '.')[-26:-3] +"._model"
        _model__layers = my_import(module_name)

    return _model__layers()

def extract_model_layers_sizes(model_path: str, layer_type: str = 'Gemm'):
    """
    Extract the input and output sizes of each layer in a model
    """
    try:
        model = onnx.load(model_path)
    except FileNotFoundError:
        print(f"Error: The file '{model_path}' was not found.")
        return []

    layer_dimensions = []
    initializer_tensors = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type == layer_type:
            print("Found Gemm node")
            weight_tensor_name = node.input[1]
            
            if weight_tensor_name in initializer_tensors:
                weight_tensor = initializer_tensors[weight_tensor_name]
                
                shape = weight_tensor.dims
                if len(shape) == 2:
                    output_dim, input_dim = shape
                    layer_dimensions.append((input_dim, output_dim))

    return layer_dimensions
    
    