import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import functools
import os


def rho(xi, rho_0, kappa_0):
    nxi = torch.norm(xi, dim=-1)
    return rho_0 * (1 - torch.exp(- kappa_0 * nxi))


class CLFDS(nn.Module):
    def __init__(self, clf_model, reg_model, w_clf=1e-4, rho_0=0.1, kappa_0=0.001):
        super(CLFDS, self).__init__()
        self.w_clf = w_clf
        self.reg_model = reg_model
        self.clf_model = clf_model
        self.rho = functools.partial(rho, rho_0=rho_0, kappa_0=kappa_0)

    def clf_cost(self, xi, d_xi):
        _, dV = self.clf_model.forward_with_grad(xi, create_graph=True)
        dim = dV.shape[-1]
        psi = torch.matmul(dV.view(-1, 1, dim), d_xi.view(-1, dim, 1))
        psi = torch.squeeze(psi) / (torch.norm(dV, dim=-1) * torch.norm(d_xi, dim=-1) + 1e-5)
        l_all = (1 + self.w_clf) * torch.sign(psi) * (psi ** 2) / 2 + (1 - self.w_clf) * (psi ** 2) / 2
        cost = torch.sum(l_all)
        return cost

    def train_clf_virtual(self, dataset, lr=1e-3, max_epochs=1000, batch_size=100, fname="clf_model", load_if_possible=False):
        if os.path.exists(fname) and load_if_possible:
            self.clf_model = torch.load(fname)
        else:
            opt = optim.Adam(self.clf_model.parameters(), lr=lr)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(max_epochs):
                epoch_cost = 0
                count = 0
                for _, (xi, V_, w) in enumerate(dataloader):
                    opt.zero_grad(set_to_none=True)
                    V = self.clf_model.forward(xi)
                    cost = nn.MSELoss()(V, V_)
                    cost.backward()
                    opt.step()

                    epoch_cost += cost
                    count += 1

                if epoch % 100 == 0:
                    torch.save(self.clf_model, fname)


                epoch_cost /= count
                print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost), end='\r')

            print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost))

    def load_models(self, clf_file, reg_file):
        self.clf_model = torch.load(clf_file)
        self.reg_model = torch.load(reg_file)

    def train_clf(self, dataset, lr=1e-3, max_epochs=1000, batch_size=100, fname="clf_model", load_if_possible=False):
        if os.path.exists(fname) and load_if_possible:
            self.clf_model = torch.load(fname)
        else:
            opt = optim.Adam(self.clf_model.parameters(), lr=lr)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(max_epochs):
                epoch_cost = 0
                count = 0
                for _, (xi, d_xi) in enumerate(dataloader):
                    opt.zero_grad(set_to_none=True)
                    cost = self.clf_cost(xi, d_xi)
                    cost.backward()
                    opt.step()

                    epoch_cost += cost
                    count += 1

                if epoch % 100 == 0:
                    torch.save(self.clf_model, fname)

                epoch_cost /= count
                print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost), end='\r')

            print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost))

    def collect_ds_data(self, dataset):
        xi = dataset.X
        d_xi = dataset.dX
        _, dV = self.clf_model.forward_with_grad(xi)
        dim = dV.shape[-1]

        psi = torch.matmul(dV.view(-1, 1, dim), d_xi.view(-1, dim, 1))
        psi = torch.squeeze(psi) + self.rho(xi)
        psi = psi / (torch.norm(dV, dim=-1) * torch.norm(dV, dim=-1) + 1e-5)
        psi = torch.unsqueeze(psi, dim=-1)
        psi = torch.repeat_interleave(psi, dim, dim=-1)
        u = - psi * dV * (psi > 0)
        dX = d_xi + u
        dataset.dX = dX.detach()
        return dataset

    def train_ds(self, dataset, lr=1e-3, max_epochs=1000, batch_size=100, fname='ds_model', load_if_possible=True):
        if isinstance(self.reg_model, nn.Module):
            if os.path.exists(fname) and load_if_possible:
                self.reg_model = torch.load(fname)
            else:
                ds_dataset = self.collect_ds_data(dataset)
                opt = optim.Adam(self.reg_model.parameters(), lr=lr)
                dataloader = DataLoader(ds_dataset, batch_size=batch_size, shuffle=True)
                for epoch in range(max_epochs):
                    epoch_cost = 0
                    count = 0
                    for _, (xi, d_xi) in enumerate(dataloader):
                        opt.zero_grad(set_to_none=True)
                        d_xi_ = self.reg_model.forward(xi)
                        cost = nn.MSELoss()(d_xi_, d_xi)
                        cost.backward()
                        opt.step()

                        epoch_cost += cost
                        count += 1

                    if epoch % 100 == 0:
                        torch.save(self.reg_model, fname)

                    epoch_cost /= count
                    print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost), end='\r')

                print('epoch: %1d / %1d, cost: %.8f' % (epoch, max_epochs, epoch_cost))

    def forward(self, x):
        return self.reg_model.forward(x)
