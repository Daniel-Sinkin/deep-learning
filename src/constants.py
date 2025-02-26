from enum import StrEnum, auto

from .activation_function import IdentityActivation, ReLUActivation, SigmoidActivation


class E_ActivationFunctionNames(StrEnum):
    RELU = auto()
    SIGMOID = auto()
    IDENTITY = auto()


activation_function_map = {
    E_ActivationFunctionNames.RELU: ReLUActivation,
    E_ActivationFunctionNames.SIGMOID: SigmoidActivation,
    E_ActivationFunctionNames.IDENTITY: IdentityActivation,
}
