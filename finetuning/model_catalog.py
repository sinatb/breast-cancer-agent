import ModelAdapter

class ModelData(object):
    def __init__(self ,name:str ,params:float, gflops:float, acc:float, adapter:ModelAdapter):
        self.params = params
        self.gflops = gflops
        self.acc = acc
        self.name= name
        self.adapter = adapter
    def __str__(self):
        return f"Model name: {self.name}, accuracy: {self.acc}, params: {self.params}, gflops: {self.gflops}"

class ModelCatalog(object):
    def __init__(self):
        super(ModelCatalog, self).__init__()
        self.model_dict = {}
    def add_module(self, name:str, params:float, gflops:float, acc:float, adapter:ModelAdapter):
        self.model_dict[name] = ModelData(name, params, gflops, acc, adapter)
    def __str__(self):
        model_str = ""
        for name, model in self.model_dict.items():
            model_str += f"{model}\n"
        return f"Models:\n{model_str}"

def get_default_catalog():
    default_catalog = ModelCatalog()
    default_catalog.add_module(name="vit_b_16",
                               acc=81.072,
                               params=86.6,
                               gflops=17.56,
                               adapter=ModelAdapter.VitAdapter)

    default_catalog.add_module(name="swin_s",
                               acc=83.196,
                               params=49.6,
                               gflops=8.74,
                               adapter=ModelAdapter.SwinAdapter)

    default_catalog.add_module(name="resnet50",
                               acc=76.13,
                               params=25.6,
                               gflops=4.09,
                               adapter=ModelAdapter.ResNetAdapter)

    default_catalog.add_module(name="mobilenet_v3_large",
                               acc=74.042,
                               params=5.5,
                               gflops=0.22,
                               adapter=ModelAdapter.MobileNetV3Adapter)

    default_catalog.add_module(name="densenet161",
                               acc=77.138,
                               params=28.7,
                               gflops=7.73,
                               adapter=ModelAdapter.DenseNetAdapter)

    return default_catalog
