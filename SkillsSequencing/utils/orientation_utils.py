from typing import Union, Tuple
import numpy as np
import torch


def sphere_logarithmic_map_batch(x, x0):
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
    This function works for batch of data

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=0)
    if len(x0.shape) < 2:
        x0 = np.expand_dims(x0, axis=0)

    x = torch.from_numpy(x)
    x0 = torch.from_numpy(x0)
    x = torch.unsqueeze(x, -1)
    x0 = torch.unsqueeze(x0, -1)

    distance = torch.acos(torch.clamp(torch.matmul(torch.transpose(x0, 1, 2), x), -1., 1.))
    x0 = torch.squeeze(x0, dim=-1)
    x = torch.squeeze(x, dim=-1)
    distance = torch.squeeze(distance).view(x0.shape[0], 1)

    u = torch.where(distance > 1e-16, (x - x0 * torch.cos(distance)) * distance / torch.sin(distance), torch.zeros((1,x.shape[-1])))
    return u.numpy()


def sphere_derivative_logarithmic_map(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    This function computes the derivative of the logarithmic map with respect to x, i.e., dLog_x0(x) / dx.

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x is mapped with the logarithmic map

    Returns
    -------
    :return: derivative of logarithmic map dLog_x0(x) / dx
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    dim = x.shape[0]

    distance = np.arccos(np.clip(np.dot(x0.T, x), -1., 1.))

    if distance < 1e-6:
        return np.zeros((dim, dim))

    else:
        projection = x - np.cos(distance)*x0
        norm_projection = np.linalg.norm(projection)
        id_matrix = np.eye(dim)

        derivative = distance * (id_matrix - np.dot(x0, x0.T)) * \
                     (id_matrix/norm_projection - np.dot(projection, projection.T)/norm_projection**3) \
                     - np.dot(x0, (projection/norm_projection).T) / np.sqrt(1.-np.cos(distance)**2)

        return derivative


def rotation_matrix_to_unit_sphere(R: np.ndarray) -> Union[np.ndarray, int]:
    """
    This function transforms a rotation matrix to a point lying on a sphere (i.e., unit vector).
    This function is valid for rotation matrices of dimension 2 (to S1) and 3 (to S3).

    Parameters
    ----------
    :param R: rotation matrix

    Returns
    -------
    :return: a unit vector on S1 or S3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if R.shape[0] == 3:
        return rotation_matrix_to_quaternion(R)
    elif R.shape[0] == 2:
        return R[:, 0]
    else:
        return -1


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    This function transforms a 3x3 rotation matrix into a quaternion.
    This function was implemented based on Peter Corke's robotics toolbox.

    Parameters
    ----------
    :param R: 3x3 rotation matrix

    Returns
    -------
    :return: a quaternion [scalar term, vector term]
    """

    qs = min(np.sqrt(np.trace(R) + 1)/2.0, 1.0)
    kx = R[2, 1] - R[1, 2]   # Oz - Ay
    ky = R[0, 2] - R[2, 0]   # Ax - Nz
    kz = R[1, 0] - R[0, 1]   # Ny - Ox

    if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]) :
        kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1 # Nx - Oy - Az + 1
        ky1 = R[1, 0] + R[0, 1]               # Ny + Ox
        kz1 = R[2, 0] + R[0, 2]               # Nz + Ax
        add = (kx >= 0)
    elif (R[1, 1] >= R[2, 2]):
        kx1 = R[1, 0] + R[0, 1]               # Ny + Ox
        ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1 # Oy - Nx - Az + 1
        kz1 = R[2, 1] + R[1, 2]               # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2, 0] + R[0, 2]               # Nz + Ax
        ky1 = R[2, 1] + R[1, 2]               # Oz + Ay
        kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1 # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.zeros(4)
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s*np.array([kx, ky, kz])
        q = np.hstack((qs, qv))

    return q


def compute_analytical_orientation_jacobian_sphere_batch(xo, orientation_jacobian, pos_dim=2):
    """
    Batch version of compute_analytical_orientation_jacobian_sphere

    Parameters
    ----------
    :param xo: orientation represented as unit vector
    :param orientation_jacobian: orientation part of the geometric Jacobian

    Returns
    -------
    :return: orientation part of the analytical Jacobian
    """
    # 2D case(circle, S1)

    if pos_dim == 2:
        import torch
        xo = torch.from_numpy(xo)
        ojacob = torch.from_numpy(orientation_jacobian)
        trans_mat = torch.stack([-xo[:,1], xo[:,0]], dim=1)
        trans_mat = torch.unsqueeze(trans_mat, -1)
        res = torch.matmul(trans_mat, ojacob)
        return res.numpy()
    else:
        xo = xo.squeeze()
        trans_mat = np.array([[-xo[1], -xo[2], -xo[3]],
                              [xo[0], xo[3], -xo[2]],
                              [-xo[3], xo[0], xo[1]],
                              [xo[2], -xo[1], xo[0]]])

        orientation_jacobian = orientation_jacobian.squeeze()
        jac = np.dot(trans_mat, orientation_jacobian)
        return np.expand_dims(jac, axis=0)
