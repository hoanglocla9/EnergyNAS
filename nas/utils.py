import nni
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