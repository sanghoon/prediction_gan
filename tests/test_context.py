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


def test_lookahead():
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


def test_lookahead2():
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
    test_lookahead()
    test_lookahead2()