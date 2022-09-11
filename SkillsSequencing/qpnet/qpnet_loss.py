import numpy as np
import torch
import torch.nn as nn
from SkillsSequencing.utils.matrices_processing import fill_diag
from torch import Tensor

from SkillsSequencing.utils.utils import prepare_torch
device = prepare_torch()


class MultipleSkillsVelocityKktLoss(nn.Module):
    """
    Instances of this class define a loss function based on the KKT optimality conditions of a QP optimization.
    Namely the loss is equal to the norm of the weighted errors, where the weight matrix is given by the QP.
    """
    def __init__(self):
        """
        Initialization of the MultipleSkillsVelocityKktLoss class.
        """
        super().__init__()

    def forward(self, input_velocity: Tensor, target_velocity: Tensor, qp_weight_matrix: Tensor,
                constant_skill_weight=None) -> Tensor:
        """
        This function computes the loss function.

        Parameters
        ----------
        :param input_velocity: current velocity of the different skills     (batch x total dim of skills)
        :param target_velocity: desired velocity                            (batch x total dim of skills)
        :param qp_weight_matrix: current QP weight in the form of a matrix  (batch x total dim of skills x total dim of skills)

        Optional parameters
        -------------------
        :param constant_skill_weight: additional weights to balance the magnitude of the different skills   (total dim of skills)

        Returns
        -------
        :return: KKT loss

        Note: We can directly velocities dx from all skills because the difference is computed from the SAME position x.
              If the position differ, we need to parallel transport the velocities to the tangent space of a single
              position to be able to compare them.
        """
        if constant_skill_weight is None:
            artificial_weight_matrix = torch.eye(qp_weight_matrix.shape[-1]).to(device)
        else:
            artificial_weight_matrix = fill_diag(constant_skill_weight)

        artificial_weight_matrix = artificial_weight_matrix.to(device)
        error = (input_velocity - target_velocity)[:, :, None]
        
        loss = torch.mean(torch.norm(torch.matmul(torch.matmul(artificial_weight_matrix, qp_weight_matrix),
                                                 torch.matmul(artificial_weight_matrix, error)), dim=1))

        return loss
