from .module import Module


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def get_parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.get_parameters())
        return params
