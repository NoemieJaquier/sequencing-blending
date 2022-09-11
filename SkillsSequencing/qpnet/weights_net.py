import numpy as np
from qpth.qp import QPFunction
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn

from SkillsSequencing.utils.matrices_processing import fill_tril
from SkillsSequencing.utils.utils import prepare_torch

device = prepare_torch()


class FullyConnectedNet(nn.Module):
    """
    Instances of this class are neural networks with one fully-connected layer. These networks are used to compute the
    parameters of a following OptNet layer based on given input data.
    """
    def __init__(self, in_dim, out_dim, out_act=None, n_layers=None):
        """
        Initialization of the FullyConnectedNet class.

        Parameters
        ----------
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param out_act: output activation function (if None, it will use linear layer).
        :param n_layers: dimension of the fully-connected layer
        """
        super(FullyConnectedNet, self).__init__()
        if n_layers is None:
            n_layers = [20, 20, out_dim]
        else:
            n_layers.append(out_dim)

        layers = []
        in_feat = in_dim
        for i in range(len(n_layers)):
            out_feat = n_layers[i]
            layer = nn.Linear(in_feat, out_feat)
            for param in layer.parameters():
                nn.init.uniform_(param, -1/in_feat, 1/in_feat)
                # nn.init.uniform_(param, -.1, .1)

            layers.append(layer)
            if i < len(n_layers) - 1:
                layers.append(nn.SiLU())

            in_feat = out_feat

        if out_act is not None:
            layers.append(out_act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        This function computes the forward pass throughout the network

        Parameters
        ----------
        :param x: input data

        Returns
        -------
        :return: output of the FullyConnectedNet given x
        """
        return self.model(x)


class PSDNet(nn.Module):
    """
    Instances of this class are neural networks generating SPD matrices. These networks are used to compute the
    parameters of a following OptNet layer based on given input data.
    """
    def __init__(self, in_dim, out_dim, out_act=None, n_layers=None):
        """
        Initialization of the FullyConnectedNet class.

        Parameters
        ----------
        :param in_dim: input dimension
        :param out_dim: output dimension, side size of the SPD matrix
        :param out_act: output activation function (if None, it will use linear layer).
        :param n_layers: dimension of the fully-connected layer
        """
        super(PSDNet, self).__init__()

        out_dim_chol = int(out_dim + out_dim * (out_dim - 1) / 2)

        if n_layers is None:
            n_layers = [20, 20, out_dim_chol]

        layers = []
        in_feat = in_dim
        for i in range(len(n_layers)):
            out_feat = n_layers[i]
            layer = nn.Linear(in_feat, out_feat)
            for param in layer.parameters():
                nn.init.uniform_(param, -1/in_feat, 1/in_feat)

            layers.append(layer)
            if i < len(n_layers) - 1:
                layers.append(nn.LeakyReLU())

            in_feat = out_feat

        if out_act is not None:
            layers.append(out_act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        This function computes the forward pass throughout the network

        Parameters
        ----------
        :param x: input data

        Returns
        -------
        :return: output of the FullyConnectedNet given x
        """
        ldata = self.model(x)
        ldata = nn.Tanh()(ldata)

        # Build lower triangular matrix
        L = fill_tril(ldata)
        # Symmetric matrix
        Q = L + L.transpose(-1, -2)
        # SPD matrix using Cholesky decomposition
        # Q = torch.matmul(L, torch.transpose(L, -1, -2))

        # Compute the eigenvalue decomposition
        eigenvalues, eigenvectors = torch.symeig(Q, eigenvectors=True)
        # Softmax to constraint the eigenvalue of the matrix, i.e, norm(Q, fro) = 1
        eigenvalues = torch.sqrt(nn.Softmax(-1)(torch.exp(eigenvalues)**2))
        Q = torch.matmul(eigenvectors, torch.matmul(torch.diag_embed(eigenvalues), torch.inverse(eigenvectors)))

        return Q


class SymmetricNet(nn.Module):
    """
    Instances of this class are neural networks generating symmetric matrices. These networks are used to compute the
    parameters of a following OptNet layer based on given input data.
    """
    def __init__(self, in_dim, out_dim, out_act=None, n_layers=None):
        """
        Initialization of the FullyConnectedNet class.

        Parameters
        ----------
        :param in_dim: input dimension
        :param out_dim: output dimension, side size of the SPD matrix
        :param out_act: output activation function (if None, it will use linear layer).
        :param n_layers: dimension of the fully-connected layer
        """
        super(SymmetricNet, self).__init__()

        out_dim_chol = int(out_dim + out_dim * (out_dim - 1) / 2)

        if n_layers is None:
            n_layers = [20, 20, out_dim_chol]

        layers = []
        in_feat = in_dim
        for i in range(len(n_layers)):
            out_feat = n_layers[i]
            layer = nn.Linear(in_feat, out_feat)
            for param in layer.parameters():
                nn.init.uniform_(param, -1/in_feat, 1/in_feat)

            layers.append(layer)
            if i < len(n_layers) - 1:
                layers.append(nn.LeakyReLU())

            in_feat = out_feat

        if out_act is not None:
            layers.append(out_act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        This function computes the forward pass throughout the network

        Parameters
        ----------
        :param x: input data

        Returns
        -------
        :return: output of the FullyConnectedNet given x
        """
        ldata = self.model(x)
        ldata = nn.Tanh()(ldata)

        # Build lower triangular matrix
        L = fill_tril(ldata, dim=int((np.sqrt(8 * ldata.shape[-1] + 1) -1)/2))
        # Symmetric matrix
        Q = L + L.transpose(-1, -2)

        return Q


class OneLayerOptNet(nn.Module):
    """
    Instances of this class are neural networks composed by an OptNet layer.
    The OptNet layer solves a QP of the form: argmin_x  0.5*x'Qx + p'x subject to Gx <= h and Ax = b.
    """
    def __init__(self, ndim, nineq=0, neq=0, eps=1e-4, Q=None, q=None, A=None, b=None, G=None, h=None):
        """
        Initialization of the OneLayerOptNet class.

        Parameters
        ----------
        :param ndim: number of dimensions of the problem

        Optional parameters
        -------------------
        :param nineq: number of inequality contraints
        :param neq: number of equality contraints
        :param eps: regularization term
        :param Q: QP parameter
        :param q: QP parameter
        :param A: QP equality constraintsparameter
        :param b: QP equality constraints parameter
        :param G: QP inequality constraints parameter
        :param h: QP inequality constraints parameter
        """
        super(OneLayerOptNet, self).__init__()
        self.ndim = ndim
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        if Q is None:
            self.M = Variable(torch.tril(torch.ones(self.ndim, self.ndim)))
            self.L = Parameter(torch.tril(torch.rand(self.ndim, self.ndim)))
            self.learn_Q = True
        else:
            self.Q = Q
            self.learn_Q = False

        if q is None:
            self.q = Parameter(torch.Tensor(self.ndim))
            nn.init.uniform_(self.q, -1, 1)
        else:
            self.q = q

        if G is None:
            self.G = Parameter(torch.Tensor(self.nineq, self.ndim))
            nn.init.uniform_(self.G, -1, 1)
        else:
            self.G = G

        if A is None:
            self.A = Parameter(torch.Tensor(self.neq, self.ndim))
            nn.init.uniform_(self.A, -1, 1)
        else:
            self.A = A

        if b is None:
            self.b = Parameter(torch.Tensor(self.neq))
            nn.init.uniform_(self.b, -1, 1)
        else:
            self.b = b

        if h is None:
            self.x0 = Parameter(torch.zeros(self.ndim))
            self.s0 = Parameter(torch.ones(self.nineq))
            self.learn_h = True
        else:
            self.h = h
            self.learn_h = False

    def forward(self):
        """
        This function computes the forward pass throughout the network

        Returns
        -------
        :return: output of the OneLayerOptNet, i.e., QP solution.
        """
        if self.learn_Q:
            L = self.M * self.L
            self.Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.ndim)).to(device)

        Q = self.Q
        G = self.G
        A = self.A
        b = self.b
        q = self.q
        if self.learn_h:
            h = self.G.mv(self.z0) + self.s0
        else:
            h = self.h

        x = QPFunction(verbose=False)(Q, q, G, h, A, b)
        return x
