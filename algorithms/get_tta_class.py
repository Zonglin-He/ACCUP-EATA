from pre_train_model.pre_train_model import PreTrainModel
from .accup import ACCUP

def get_algorithm_class(algorithm_name): # 根据给定的算法名称字符串返回对应的算法类
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
