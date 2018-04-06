import torch
import torch.nn as nn
from torch.autograd import Variable
from modules import Prediction
import numpy as np


class MockNet(nn.Module):
    def __init__(self):
        super(MockNet, self).__init__()
        self.fc = nn.Linear(3, 16)

    def forward(self, x):
        x = self.fc(x)

        return x


def test_context():
    net = MockNet()

    input = Variable(torch.randn((5, 3)))

    # Zero-weights, zero-biases
    net.fc.weight.data.fill_(0.0)
    net.fc.bias.data.fill_(0.0)

    pred = Prediction(net, clone_model=False)

    # Update weights
    net.fc.weight.data.normal_(0, 1.0)

    result1 = net(input)

    with pred.peek(1.0):
        result2 = net(input)

    with pred.peek(2.0):
        result3 = net(input)

    result4 = net(input)


    assert (np.all(np.isclose(result1.data.numpy(), result2.data.numpy())))
    assert (np.all(np.isclose((2.0 * result1).data.numpy(), result3.data.numpy())))
    assert (np.all(np.isclose(result1.data.numpy(), result4.data.numpy())))


if __name__ == "__main__":
    test_context()