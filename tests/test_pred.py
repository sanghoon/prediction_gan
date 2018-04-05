import torch
import torch.nn as nn
from torch.autograd import Variable
from modules import PredictionModule
import numpy as np


class MockNet(nn.Module):
    def __init__(self):
        super(MockNet, self).__init__()
        self.fc = nn.Linear(3, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.fc2(x)

        return x


def test_if_copied():
    net = MockNet()

    net.fc.weight.data.fill_(1.0)
    net.fc.bias.data.fill_(2.0)

    pred = PredictionModule(net)

    # Check whether they have the same values
    assert(np.all(pred.module.fc.weight.data == net.fc.weight.data))
    assert (np.all(pred.module.fc.bias.data == net.fc.bias.data))

    net.fc.weight.data.fill_(3.0)

    # Check whether they are seperate variables
    assert (np.all(pred.module.fc.weight.data != net.fc.weight.data))


def test_prediction():
    net = MockNet()

    net.fc.weight.data.fill_(1.0)
    net.fc.bias.data.fill_(2.0)

    pred = PredictionModule(net)

    pred.update_copy()

    # No changes yet
    assert (np.all(pred.module.fc.weight.data == net.fc.weight.data))

    net.fc.weight.data.fill_(2.0)           # Increase by 1.0

    pred.update_copy()

    assert (np.all(pred.module.fc.weight.data == 3.0))      # Increased by 2 * 1.0

    # No changes in the original module
    assert (np.all(net.fc.weight.data == 2.0))


def test_bn():
    net = MockNet()

    net.bn.weight.data.fill_(1.0)
    net.bn.running_mean.fill_(1.)

    pred = PredictionModule(net)

    net.bn.weight.data.fill_(2.0)           # Increase weight by 1.0
    net.bn.running_mean.fill_(2.0)     # Increase running_mean by 1.0

    pred.update_copy()

    # Weight should change, but running_mean shouldn't
    assert (np.all(pred.module.bn.weight.data == 3.0))
    assert (np.all(pred.module.bn.running_mean == 1.0))


def test_integrity():
    net = MockNet()
    net.fc.weight.data.normal_(0.0, 1.0)

    pred = PredictionModule(net)

    input = Variable(torch.randn((5, 3)))

    assert(np.all(net(input).data.numpy() == pred(input).data.numpy()))

    # No changes
    pred.update_copy(step=1.0)

    assert (np.all(net(input).data.numpy() == pred(input).data.numpy()))

    # New weights
    net.fc.weight.data.normal_(0.0, 1.0)
    pred.update_copy(step=1.0)

    assert (np.all(np.isclose(net(input).data.numpy(), pred(input).data.numpy())))

    # New weights
    net.fc.weight.data.normal_(0.0, 1.0)
    pred.update_copy(step=2.0)

    assert (np.any(np.not_equal(net(input).data.numpy(), pred(input).data.numpy())))


def test_grad_opt():
    net = MockNet()
    pred = PredictionModule(net)

    assert(net.fc.weight.requires_grad)
    assert(not pred.module.fc.weight.requires_grad)


if __name__ == "__main__":
    test_if_copied()
    test_prediction()
    test_bn()
    test_integrity()
    test_grad_opt()