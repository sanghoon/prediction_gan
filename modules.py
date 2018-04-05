import torch.nn as nn
import copy


class PredictionModule(nn.Module):
    def __init__(self, model):
        """
        :param model:   nn.Module
        """
        super(PredictionModule, self).__init__()

        self._org_model = model
        self.module = copy.deepcopy(model)

        # Parameters
        self._last_copy = copy.deepcopy(dict(model.named_parameters()))

        for k, p in self._last_copy.items():
            p.requires_grad = False

        for k, p in self.module.named_parameters():
            p.requires_grad = False

    def step(self, step=2.0):
        new_copy = dict(self._org_model.named_parameters())
        my_state = dict(self.module.named_parameters())

        for k, src in new_copy.items():
            assert(k in self._last_copy.keys())
            assert(k in my_state.keys())

            old = self._last_copy[k]
            dst = my_state[k]

            # Update params. for inference
            dst.data[:] = step * (src.data[:] - old.data[:]) + old.data[:]

            # Update params. to calculate delta
            old.data[:] = src.data[:]


        # Copy other params. (TODO: better impl.?)
        src_modules = list(self._org_model.modules())
        dst_modules = list(self.module.modules())

        for s, d in zip(src_modules, dst_modules):
            assert(type(s) == type(d))

            # Handle BN
            if (isinstance(s, nn.BatchNorm1d) or isinstance(s, nn.BatchNorm2d) or isinstance(s, nn.BatchNorm3d)):
                d.running_mean[:] = s.running_mean[:]
                d.running_var[:] = s.running_var[:]


    def forward(self, x):
        return self.module(x)