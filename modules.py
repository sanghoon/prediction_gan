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
            p.required_grad = False

        for k, p in self.module.named_parameters():
            p.required_grad = False

    def update_copy(self):
        new_copy = self._org_model.named_parameters()
        my_state = dict(self.module.named_parameters())

        for k, p in new_copy:
            assert(k in self._last_copy.keys())
            assert(k in my_state.keys())

            # Update params. for inference
            my_state[k].data[:] = (2 * p.data[:] - self._last_copy[k].data[:])      # P_new + (P_new - P_old)

            # Update params. to calculate delta
            self._last_copy[k].data[:] = p.data[:]

    def forward(self, x):
        return self.module(x)