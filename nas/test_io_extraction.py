import onnx

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

if __name__ == '__main__':
    network_4 = "/home/jurianonderwater/Documents/Code/STM-REST/STM32EdgeAI-REST/src/generate_models/models/234_245_140_10_246.onnx"
    sizes = extract_model_layers_sizes(network_4)
    for layer in sizes:
        print(layer)
    print(len(sizes))