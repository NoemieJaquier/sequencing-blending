import numpy as np
import torch
import torch.nn as nn
from SkillsSequencing.utils.utils import prepare_torch
device = prepare_torch()


class Constraint(nn.Module):
    """
    Instances of this class define constraints for an OptNet layer.
    """
    def __init__(self):
        super(Constraint, self).__init__()
        self.set_trainable(False)

    def update(self, **kwargs):
        pass

    def set_trainable(self, trainable=False):
        """
        Set parameters of the constraints as trainable or not.

        Parameters
        ----------
        :param trainable: True or False

        Returns
        -------
        :return: -
        """
        for param in self.parameters():
            if trainable:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def print(self):
        pass


class EqnConstraint(Constraint):
    """
    Instances of this class define equality constraints for an OptNet layer. Constraints are of the type Ax = b.
    """
    def __init__(self, A=None, b=None):
        """
        Initialization of the EqnConstraint class.

        Parameters
        ----------
        :param A: constraint parameter Ax = b
        :param b: constraint parameter Ax = b
        """
        super(EqnConstraint, self).__init__()
        if A is None or b is None:
            A = torch.autograd.Variable(torch.Tensor())
            b = torch.autograd.Variable(torch.Tensor())

        self.A = A.to(device)
        self.b = b.to(device)
        self.set_trainable(False)

    def print(self):
        print('A is {}, b is {}'.format(self.A.cpu().numpy(), self.b.cpu().numpy()))

    def update(self, q):
        pass


class IneqnConstraint(Constraint):
    """
    Instances of this class define inequality constraints for an OptNet layer. Constraints are of the type Gx <= h.
    """
    def __init__(self, dim, G=None, h=None):
        """
        Initialization of the IneqnConstraint class.

        Parameters
        ----------
        :param dim: dimension of the constraint @YOU: why do we need a dimension here and not for the equality constraint?
        :param G: constraint parameter Gx <= h
        :param h: constraint parameter Gx <= h
        """
        super(IneqnConstraint, self).__init__()
        if G is None or h is None:
            # TODO: solve: QPFunction bugs when a zero-dimensional G is given (line 87 of qpth/qp.py)
            # TODO: for now we give one always-valid constraint Gx <= h, where G=0 and h=0
            G = torch.from_numpy(np.array([[0]*dim])).double()
            h = torch.from_numpy(np.array([[0]])).double()
            # G = torch.autograd.Variable(torch.Tensor())
            # h = torch.autograd.Variable(torch.Tensor())

        self.G = G.to(device)
        self.h = h.to(device)
        self.dim = dim
        self.set_trainable(False)

    def print(self):
        print('G is {}, h is {}'.format(self.G.cpu().numpy(), self.h.cpu().numpy()))

    def update(self, q):
        pass


class ListEqnConstraints(Constraint):
    """
    Instances of this class combines a list of equality constraints for an OptNet layer into one constraint Ax = b.
    """
    def __init__(self, eq_constraints_list):
        """
        Initialization of the ListEqnConstraints class.

        Parameters
        ----------
        :param eq_constraints_list: list of inequality constraints of the type EqnConstraint.
        """
        super(ListEqnConstraints, self).__init__()

        # Initialize constraints list
        self.eq_constraints_list = eq_constraints_list
        self.n_constraints = len(self.eq_constraints_list)

        # Initialize constraint parameters
        self.A = self.eq_constraints_list[0].A
        self.B = self.eq_constraints_list[0].b
        for n in range(1, self.n_constraints):
            self.A = torch.cat([self.A, self.eq_constraints_list[n].A], dim=0)
            self.b = torch.cat([self.b, self.eq_constraints_list[n].b], dim=0)

    def print(self):
        print('A is {}, b is {}'.format(self.A.cpu().numpy(), self.b.cpu().numpy()))


class ListIneqnConstraints(Constraint):
    """
    Instances of this class combines a list of inequality constraints for an OptNet layer into one constraint Gx <= h.
    """
    def __init__(self, ineq_constraints_list):
        """
        Initialization of the ListIneqnConstraints class.

        Parameters
        ----------
        :param ineq_constraints_list: list of inequality constraints of the type IneqnConstraint.
        """
        super(ListIneqnConstraints, self).__init__()

        # Initialize constraints list
        self.ineq_constraints_list = ineq_constraints_list
        self.n_constraints = len(self.ineq_constraints_list)

        # Initialize constraint parameters
        self.G = self.ineq_constraints_list[0].G
        for n in range(1, self.n_constraints):
            self.G = torch.cat([self.G, self.ineq_constraints_list[n].G], dim=0)
        self.h = None

    def update(self, q):
        """
        This function updates the constraints of the list given the current joint position of the robot.

        Parameters
        ----------
        :param q: current joint angles

        Returns
        -------
        :return: -
        """
        # Update constraints
        for constraint in self.ineq_constraints_list:
            constraint.update(q)
        # Update G and h
        G = self.ineq_constraints_list[0].G
        h = self.ineq_constraints_list[0].h
        for n in range(1, self.n_constraints):
            G = torch.cat([G, self.ineq_constraints_list[n].G], dim=0)
            h = torch.cat([h, self.ineq_constraints_list[n].h], dim=1)
        self.G = G.to(device)
        self.h = h.to(device)

    def print(self):
        print('G is {}, h is {}'.format(self.G.cpu().numpy(), self.h.cpu().numpy()))


class JointAngleLimitsConstraint(IneqnConstraint):
    """
    Instances of this class defines joint angles inequality constraints for an OptNet layer.
    """
    def __init__(self, dim, angle_max=np.pi-0.2, angle_min=-np.pi+0.2, dt=0.01):
        """
        Initialization of the JointAngleLimitsConstraint class.

        Parameters
        ----------
        :param dim: number of DoFs of the robot
        :param angle_max: maximum joint angle
        :param angle_min: minimum joint angle
        :param dt: time step
        """
        super(JointAngleLimitsConstraint, self).__init__(dim)

        self.dt = dt

        if np.isscalar(angle_max):
            self.angle_max = angle_max
            self.angle_min = angle_min
        else:
            while len(angle_max) < dim:
                angle_max = np.append(angle_max, 0.0)
                angle_min = np.append(angle_min, 0.0)

            self.angle_max = torch.from_numpy(angle_max)
            self.angle_min = torch.from_numpy(angle_min)

        # Initialize parameters of the constraint Gx <= h
        G = np.vstack((np.eye(dim), -np.eye(dim)))
        self.G = torch.from_numpy(G)
        self.h = None

    def update(self, q):
        """
        This function updates the joint angles constraint parameters G and h given the current joint position of the
        robot.

        Parameters
        ----------
        :param q: current joint angles

        Returns
        -------
        :return: -
        """
        q = q.to(device)
        batch_size = q.shape[0]
        dim = q.shape[-1]

        h_max = (self.angle_max - q) / self.dt * torch.ones(size=(batch_size, dim)).to(device)
        h_min = (q - self.angle_min) / self.dt * torch.ones(size=(batch_size, dim)).to(device)

        self.h = torch.cat([h_max, h_min], dim=1)
        self.G = self.G.to(device)
        self.h = self.h.to(device)


class JointVelocityLimitsConstraint(IneqnConstraint):
    """
    Instances of this class defines joint velocity inequality constraints for an OptNet layer.
    """
    def __init__(self, dim, velocity_max=10, velocity_min=-10, dt=0.01):
        """
        Initialization of the JointVelocityLimitsConstraint class.

        Parameters
        ----------
        :param dim: number of DoFs of the robot
        :param velocity_max: maximum authorized velocity
        :param velocity_min: minimum authorized velocity
        :param dt: time step
        """
        super(JointVelocityLimitsConstraint, self).__init__(dim)

        G = np.vstack((np.eye(dim), -np.eye(dim)))
        self.G = torch.from_numpy(G)
        self.h = None
        self.dt = dt
        self.velocity_max = velocity_max
        self.velocity_min = velocity_min

    def update(self, q):
        """
        This function updates the velocity constraint parameters G and h given the current joint position of the robot.

        Parameters
        ----------
        :param q: current joint angles

        Returns
        -------
        :return: -
        """
        q = q.to(device)
        batch_size = q.shape[0]
        dim = q.shape[-1]

        h_max = self.velocity_max / self.dt * torch.ones(size=(batch_size, dim)).to(device)
        h_min = -self.velocity_min / self.dt * torch.ones(size=(batch_size, dim)).to(device)

        self.h = torch.cat([h_max, h_min], dim=1)
        self.G = self.G.to(device)
        self.h = self.h.to(device)


class TaskSpaceVariableConstraint(EqnConstraint):
    """
    Instances of this class defines task-space equality constraints for an OptNet layer. It handles the case where
    several skills influence the same task variable, therefore the components of the output of the QP corresponding to
    the same task variable have to be equal.
    """
    def __init__(self, skill_list, ctrl_idx):
        """
        Initialization of the TaskSpaceVariableConstraint class.

        Parameters
        ----------
        :param skill_list: list of skills
        """
        super(TaskSpaceVariableConstraint, self).__init__()

        self.skill_list = skill_list
        self.nb_skills = len(skill_list)

        # Skills dimensions
        self.skills_dim = np.zeros(self.nb_skills + 1, int)
        for i in range(0, self.nb_skills):
            self.skills_dim[i+1] = self.skill_list[i].dim()
        skills_dim_cumsum = np.cumsum(self.skills_dim)

        # Create constraints matrix
        A_rows = []
        head = []
        tail = []
        for i in range(self.nb_skills):
            for j in range(i+1, self.nb_skills):
                # If two skills are from the same type, the output of the QP for these skills has to be the same task
                # variable, i.e., x_i - x_j = 0, equivalently [I 0; 0 -I] * [x_i x_j] = 0
                if ctrl_idx[i] == ctrl_idx[j] and i not in head and j not in tail:
                    head.append(i)
                    tail.append(j)
                    a = np.zeros((skill_list[i].dim(), skills_dim_cumsum[-1]))
                    a[:, skills_dim_cumsum[i]:skills_dim_cumsum[i + 1]] = np.eye(skill_list[i].dim())
                    a[:, skills_dim_cumsum[j]:skills_dim_cumsum[j + 1]] = -np.eye(skill_list[j].dim())
                    A_rows.append(a)

        if len(A_rows) != 0:
            A = torch.from_numpy(np.concatenate(A_rows, axis=0))
            b = torch.zeros(A.shape[0])
            self.A = A.to(device)
            self.b = b.to(device)
        else:
            self.A = None
            self.b = None

    def update(self, x):
        """
        This function updates the constraint parameters A and b given the batch dimension of the task-space variables.

        Parameters
        ----------
        :param x: current task-space variables

        Returns
        -------
        :return: -
        """
        x = x.to(device)
        batch_size = x.shape[0]
        dim = self.A.shape[0]

        self.b = torch.zeros(size=(batch_size, dim)).to(device)


