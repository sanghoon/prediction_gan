import torch.nn as nn
import copy
from contextlib import contextmanager

# TODO: impl. w contextmanager
class Prediction(nn.Module):
    def __init__(self, model, clone_model=True):
        """
        :param model:   nn.Module
        """
        # FIXME: Module or Optim
        super(Prediction, self).__init__()

        self._org_model = model

        self.module = copy.deepcopy(model) if clone_model else model

        self._cloned = clone_model

        # Parameters
        self._modify_target = dict(self.module.named_parameters())
        self._prev_params = copy.deepcopy(dict(model.named_parameters()))

        for k, p in self._prev_params.items():
            p.requires_grad = False

        if clone_model:
            for k, p in self.module.named_parameters():
                p.requires_grad = False


    def predict_step(self, step=2.0):
        new_copy = dict(self._org_model.named_parameters())

        for k, src in new_copy.items():
            assert(k in self._prev_params.keys())
            assert(k in self._modify_target.keys())

            old = self._prev_params[k]
            dst = self._modify_target[k]

            # Update params. for inference
            dst.data[:] = step * (src.data[:] - old.data[:]) + old.data[:]

            # Update params. to calculate delta
            old.data[:] = src.data[:]

        # Copy other params. (TODO: better impl.?)
        # FIXME: is this necessary?
        src_modules = list(self._org_model.modules())
        dst_modules = list(self.module.modules())

        for s, d in zip(src_modules, dst_modules):
            assert(type(s) == type(d))

            # Handle BN
            if (isinstance(s, nn.BatchNorm1d) or isinstance(s, nn.BatchNorm2d) or isinstance(s, nn.BatchNorm3d)):
                d.running_mean[:] = s.running_mean[:]
                d.running_var[:] = s.running_var[:]

    @contextmanager
    def peek(self, step=2.0):
        assert(self.module == self._org_model)

        # Backup original params.
        new_copy = dict(self._org_model.named_parameters())
        _backup_params = copy.deepcopy(new_copy)

        for k, src in new_copy.items():
            assert(k in self._prev_params.keys())
            assert(k in self._modify_target.keys())

            old = self._prev_params[k]
            dst = src

            # Update params. for inference
            dst.data[:] = step * (src.data[:] - old.data[:]) + old.data[:]

        yield

        # Roll-back to the original params
        for k, src in new_copy.items():
            assert(k in _backup_params.keys())

            src.data[:] = _backup_params[k].data[:]

    def forward(self, x):
        if not self._cloned:
            raise RuntimeWarning("Module is not cloned. Use peek()")

        return self.module(x)
