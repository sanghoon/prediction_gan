import torch
import torch.nn as nn
import torch.optim as optim
import copy
from contextlib import contextmanager


class PredOpt(optim.Optimizer):
    def __init__(self, params):
        super(optim.Optimizer, self).__init__()

        self._params = list(params)
        self._prev_params = copy.deepcopy(self._params)
        self._diff_params = None

        self.step()

    def step(self):
        if self._diff_params is None:
            # Preserve parameter memory
            self._diff_params = copy.deepcopy(self._params)

        for i, _new_param in enumerate(self._params):
            # Calculate difference and store new params
            self._diff_params[i].data[:] = _new_param.data[:] - self._prev_params[i].data[:]
            self._prev_params[i].data[:] = _new_param.data[:]

    @contextmanager
    def lookahead(self, step=1.0):
        # Do nothing if lookahead stepsize is 0.0
        if step == 0.0:
            yield
            return

        for i, _cur_param in enumerate(self._params):
            # Integrity check (whether we have the latest copy of parameters)
            if torch.sum(_cur_param.data[:] != self._prev_params[i].data[:]) > 0:
                raise RuntimeWarning("Stored parameters differ from the current ones. Call step() after parameter updates")

            _cur_param.data[:] += step * self._diff_params[i].data[:]

        yield

        # Roll-back to the original values
        for i, _cur_param in enumerate(self._params):
            _cur_param.data[:] = self._prev_params[i].data[:]