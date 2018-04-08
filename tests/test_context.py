import torch
import torch.nn as nn
from torch.autograd import Variable
from prediction import PredOpt
import numpy as np


class MockNet(nn.Module):
    def __init__(self):
        super(MockNet, self).__init__()
        self.fc = nn.Linear(3, 16)

    def forward(self, x):
        x = self.fc(x)

        return x


def test_param_update():
    net = MockNet()

    net.fc.weight.data.fill_(0.0)
    net.fc.bias.data.fill_(0.0)

    pred = PredOpt(net.parameters())

    # Update weights
    net.fc.weight.data.fill_(1.0)                   # 0.0 => 1.0    (Increased by 1.0)
    net.fc.bias.data.fill_(0.5)                     # 0.0 => 0.5    (Increased by 0.5)

    pred.step()

    with pred.lookahead(1.0):
        assert (net.fc.weight.data[0,0] == 2.0)     # 1.0 + 1.0 * 1.0
        assert (net.fc.bias.data[0] == 1.0)         # 0.5 + 0.5 * 1.0

    assert(net.fc.weight.data[1,1] == 1.0)          # Went back to the correct value (1.0)
    assert(net.fc.bias.data[1] == 0.5)              # Went back to the correct value (1.0)

    with pred.lookahead(5.0):
        assert (net.fc.weight.data[2,2] == 6.0)     # 1.0 + 1.0 * 5.0
        assert (net.fc.bias.data[2] == 3.0)         # 0.5 + 0.5 * 5.0


def test_lookahead_pred():
    net = MockNet()

    input = Variable(torch.randn((5, 3)))

    # Zero-weights, zero-biases
    net.fc.weight.data.fill_(0.0)
    net.fc.bias.data.fill_(0.0)

    pred = PredOpt(net.parameters())

    # Update weights
    net.fc.weight.data.normal_(0, 1.0)
    pred.step()

    result1 = net(input)

    # Lookahead 0.0 => the same results
    with pred.lookahead(0.0):
        result2 = net(input)
    assert (np.all(np.isclose(result1.data.numpy(), result2.data.numpy())))

    # Lookahead 1.0 => doubled results
    with pred.lookahead(1.0):
        result3 = net(input)
    assert (np.all(np.isclose((2.0 * result1).data.numpy(), result3.data.numpy())))

    # Outside of 'with' statements => the same results
    result4 = net(input)
    assert (np.all(np.isclose(result1.data.numpy(), result4.data.numpy())))


def test_lookahead_pred2():
    net = MockNet()

    input = Variable(torch.randn((5, 3)))

    # Zero-weights, zero-biases
    net.fc.weight.data.normal_(0.0, 1.0)
    net.fc.bias.data.fill_(0.0)

    pred = PredOpt(net.parameters())

    result1 = net(input)

    # No step() yet => the same result
    with pred.lookahead(2.0):
        result2 = net(input)
    assert (np.all(np.isclose(result1.data.numpy(), result2.data.numpy())))


if __name__ == "__main__":
    test_param_update()
    test_lookahead_pred()
    test_lookahead_pred2()