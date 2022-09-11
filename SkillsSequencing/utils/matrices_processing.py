import os
import inspect
import torch

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

use_cuda = torch.cuda.is_available()
device = torch.device("cpu")  # torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)


def fill_diag(data):
    """
    This function fills a diagonal matrix with a list of values.

    Parameters
    ----------
    :param data: values to fill the diagonal matrix

    Returns
    -------
    :return: diagonal matrix
    """
    if len(data.shape) < 2:
        data = torch.unsqueeze(data, 0)

    data = data.to(device)
    batch_size = data.shape[0]
    dim = data.shape[-1]
    mask = torch.diag(torch.ones(dim)) > 0
    mask = mask.to(device)
    out = torch.zeros(size=(batch_size, dim, dim)).to(device)
    out.masked_scatter_(mask, data)
    return out


def fill_tril(data, dim, off_diag=0):
    """
    This function fills a lower-triangular matrix with a list of values.

    Parameters
    ----------
    :param data: values to fill the lower-triangular matrix

    Returns
    -------
    :return: lower-triangular matrix
    """
    if len(data.shape) < 2:
        data = torch.unsqueeze(data, 0)

    batch_size = data.shape[0]
    # dim = data.shape[-1]
    # n = int((np.sqrt(8 * dim + 1) -1)/2)

    mask = torch.tril(torch.ones(size=(batch_size, dim, dim)), diagonal=-off_diag)

    mask = mask > 0
    mask = mask.to(device)
    out = torch.zeros(size=(batch_size, dim, dim)).to(device)
    out.masked_scatter_(mask, data)
    return out


def assign_matrix(mask, target, rind, cind):
    """
    Assign the values in mask matrix to the target matrix
    :param mask: the values to be assigned
    :param target: the target matrix to be assigned
    :param rind: the row index to assign
    :param cind: the column index to assign
    :return:
    """
    if mask.shape[-1] != len(cind) or mask.shape[-2] != len(rind):
        raise ValueError('The index size does not correspond to the row or column size of the mask')

    r = 0
    c = 0
    for i in range(len(rind)):
        for j in range(len(cind)):
            target[:, rind[i], cind[j]] = mask[:, r,c]
            c += 1

        r += 1
        c = 0

    return target
