from typing import Mapping
from collections.abc import Collection
import numpy as np

class Optimizer:
    """
      This class is responsible for calculating parameter updates during optimization.
    """
    LEARNING_RATE = 1e-2

    def __init__(self, params_dict: Mapping[str, np.array]):
      """
      :param params_dict:
        A dictionary of the parameter names to be optimized.
      """
      pass

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        """
            Calculate the parameter updates for a single step of optimization.
        :param param_grads:
            A dictionary of the parameter gradients.
        :return:
          A dictionary of parameter updates.
        """
        param_updates = {}
        for param_name, param_grad in param_grads.items():
            param_updates[param_name] = self.LEARNING_RATE * param_grad

        return param_updates


class MomentumOptimizer(Optimizer):
    MOMENTUM_COEFF = 0.9

    def __init__(self, params_dict: Mapping[str, np.array]):
        super().__init__(params_dict)
        self.velocity = {}
        for param_name, param_values in params_dict.items():
            self.velocity[param_name] = np.zeros_like(param_values)

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        param_updates = {}
        for param_name, param_grad in param_grads.items():
            self.velocity[param_name] = self.MOMENTUM_COEFF * self.velocity[param_name] - self.LEARNING_RATE * param_grad
            param_updates[param_name] = self.velocity[param_name]

        return param_updates


class RMSPropOptimizer(Optimizer):
    MOMENTUM_COEFF = 0.9
    DECAY_RATE = 0.99

    def __init__(self, params_dict: Mapping[str, np.array]):
        super().__init__(params_dict)
        self.velocity = {}
        self.cache = {}
        for param_name, param_values in params_dict.items():
            self.velocity[param_name] = np.zeros_like(param_values)
            self.cache[param_name] = np.zeros_like(param_values)

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        param_updates = {}
        for param_name, param_grad in param_grads.items():
            self.cache[param_name] = self.DECAY_RATE * self.cache[param_name] + (1 - self.DECAY_RATE) * (param_grad ** 2)
            adaptive_lr = self.LEARNING_RATE / (np.sqrt(self.cache[param_name]) + 1e-8)
            param_updates[param_name] = -adaptive_lr * param_grad

        return param_updates

