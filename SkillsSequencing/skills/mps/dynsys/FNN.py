import os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, "..")
os.sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=(20,20), act="leaky_relu", out_act=None):
        super(SimpleNN, self).__init__()
        layers = []
        in_ = in_dim
        for i in range(len(n_layers)):
            out_ = n_layers[i]
            layers.append(nn.Linear(in_, out_))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())

            in_ = out_

        layers.append(nn.Linear(in_, out_dim))
        if out_act:
            if out_act == "relu":
                layers.append(nn.ReLU())
            elif out_act == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif out_act == "tanh":
                layers.append(nn.Tanh())

        self.layers = layers
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.fnn = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if type(layer) == nn.Linear:
                nn.init.uniform_(layer.weight, -0.0001, 0.0001)
                nn.init.uniform_(layer.bias, -0.0001, 0.0001)

    def get_cost(self, x, y, weights=None):
        y_ = self.forward(x)
        mse = nn.MSELoss()
        cost = mse(y_, y)
        return cost, y_

    def forward(self, x):
        return self.fnn(x)

    def forward_with_grad(self, input):
        input.requires_grad = True

        z = self.forward(input)
        grad = torch.autograd.grad(z, [input], grad_outputs=torch.ones_like(z), retain_graph=True)[0]
        return z, grad

    def print_struct(self):
        print(self.fnn)

    def simple_train(self, dataset, lrate=1e-3, batch_size=100, max_epochs=100, fname='simple_nn'):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optim = torch.optim.Adam(self.parameters(), lr=lrate)
        for epoch in range(max_epochs):
            epoch_cost = 0
            for i, (x, y) in enumerate(dataloader):
                y_ = self.forward(x)
                optim.zero_grad(set_to_none=True)
                cost = nn.MSELoss()(y_, y)
                cost.backward()
                optim.step()

                epoch_cost += cost

            if epoch % 20 == 0:
                torch.save(self, fname)

            print('epoch: %1d, cost: %.8f' % (epoch, epoch_cost/i), end='\r')

        print('epoch: %1d, cost: %.8f' % (epoch, epoch_cost / i), end='\r')


