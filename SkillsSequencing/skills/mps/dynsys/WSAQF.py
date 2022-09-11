import torch
import torch.nn as nn
import numpy as np

def gen_pd_mat(A):
    A = A * torch.transpose(A, -1, -2)
    A = A + A.shape[-1] * torch.eye(A.shape[-1])
    return A

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class WSAQF(nn.Module):
    def __init__(self, dim=2, n_qfcn=2):
        super(WSAQF, self).__init__()
        self.n_qfcn = n_qfcn
        self.Ps = nn.ParameterList()
        self.Mus = nn.ParameterList()
        for i in range(n_qfcn+1):
            P = nn.parameter.Parameter(torch.Tensor(dim, dim))
            nn.init.uniform_(P, -0.001, 0.001)
            self.Ps.append(P)

            Mu = nn.parameter.Parameter(torch.Tensor(dim,1))
            nn.init.uniform_(Mu, -0.001, 0.001)
            self.Mus.append(Mu)

    def forward(self, xi):
        xi = torch.unsqueeze(xi, dim=-1)
        P0 = gen_pd_mat(torch.sigmoid(self.Ps[0]))
        V = torch.matmul(torch.transpose(xi, -1, -2), torch.matmul(P0, xi))
        V = torch.squeeze(V)
        for i in range(1, self.n_qfcn+1):
            P = gen_pd_mat(torch.sigmoid(self.Ps[i]))
            component = torch.matmul(torch.transpose(xi, -1, -2), torch.matmul(P, (xi - self.Mus[i])))
            component = torch.squeeze(component)
            V = V + torch.where(component >= 0, 1, 0) * (component ** 2)

        return V

    def forward_with_grad(self, xi, create_graph=False):
        xi.requires_grad = True

        V = self.forward(xi)
        grad = torch.autograd.grad(V, [xi], grad_outputs=torch.ones_like(V), retain_graph=True, create_graph=create_graph)[0]

        return V, grad


if __name__ == '__main__':
    Vfcn = WSAQF()
    xi = torch.rand((10, 2))
    y, dy = Vfcn.forward_with_grad(xi)
    print(y)