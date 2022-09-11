import torch.optim as optim
from qpth.qp import QPFunction

from SkillsSequencing.qpnet.weights_net import FullyConnectedNet
from SkillsSequencing.qpnet.constraints import *
from SkillsSequencing.utils.matrices_processing import fill_diag
from SkillsSequencing.qpnet.qpnet_loss import MultipleSkillsVelocityKktLoss
from SkillsSequencing.utils.spd_manifold_utils import sqrtm_torch
from SkillsSequencing.utils.utils import prepare_torch

device = prepare_torch()

opt_policies = ['SkillDiagonalWeightedPolicy', 'SkillFullyWeightedPolicy']
non_opt_policies = []


def copy_policy_parameters(current_policy, new_policy, copy_skills=False, copy_constraints=False):
    """
    This function copies the parameters from a given (current) policy onto a new policy. This is typically used to train
    a fully-weighted-matrix-based policy, which is initialized with the parameter of a previously-trained
    diagonal-matrix-based policy.

    Parameters
    ----------
    :param current_policy: policy to copy the parameters from
    :param new_policy: policy to copy the parameters to
    :param copy_skills: if True, the skills of the current policy are also copied to the new policy
    :param copy_constraints: if True, the constraints of the current policy are also copied to the new policy

    Returns
    -------
    :return: -
    """

    if hasattr(current_policy, 'Qnet'):
        new_policy.to(device)
        new_policy.Qnet = current_policy.Qnet
    elif hasattr(current_policy, 'QDiagNet'):
        new_policy.to(device)
        new_policy.QDiagNet = current_policy.QDiagNet
        new_policy.QOffDiagNet = current_policy.QOffDiagNet
    else:
        raise ValueError("cannot find a joint space policy that is compatible with the task space one")

    new_policy.constant_skill_weights = current_policy.constant_skill_weights
    new_policy.eps = current_policy.eps
    new_policy.dt = current_policy.dt

    if copy_skills:
        new_policy.skills = current_policy.skills

    if copy_constraints:
        new_policy.eqn = current_policy.eqn
        new_policy.ineqn = current_policy.ineqn


class SkillPolicy(nn.Module):
    """
    This class is the base class to policies sequencing and blending skills using the differentiable-optimization layer
    Optnet. The class is given a list of skills with corresponding equality and inequality constraints that are applied
    to the control values of the skills. For example, the skills whose control values apply to the same quantity (e.g.,
    joint velocities) should be given an equality constraints. The policy can also be provided with constant skills
    weights which aims at balancing the different of magnitude between the control values of the different skills.
    This class is used as a template for all other policies.
    """
    def __init__(self, dim, fdim, skill, eqn=None, ineqn=None, constant_skill_weights=None, eps=1e-10, dt=0.01):
        """
         Initialization of the SkillPolicy class.

         Parameters
         ----------
         :param dim: TODO can we remove this parameter?
         :param fdim: the dimension of the features for the parameters of the QP
         :param skills: a list of skills

         Optional parameters
         -------------------
         :param eqn: equality constraints of the form Ax = b
         :param ineqn: inequality constraints of the form Gx < h
         :param eps: regularization value
         :param dt: time step
         """
        super(SkillPolicy, self).__init__()
        self.dim = dim
        self.fdim = fdim
        # Initialize the constraints
        if eqn is None:
            eqn = EqnConstraint()

        if ineqn is None:
            ineqn = IneqnConstraint(dim)

        self.eqn = eqn
        self.ineqn = ineqn

        # List of skills
        self.skill = skill

        # Initialize magnitude of skills (used to account for important differences in magnitude)
        self.constant_skill_weights = constant_skill_weights

        # Skills are a list of skills
        self.eps = eps
        self.dt = dt

        self.use_state_space = True
        # Initialize QP parameters
        self.Q = None
        self.wmat = None
        self.is_train = True

    def get_skill_mat(self, x, desired_):
        """
        This function computes skill vectors and matrices that are used in the SkillPolicy classes to obtain the
        parameters Q and p of the QP problem, so that the problem has the standard from x'Qx + p'x.
        To do so, the function updates the desired values of the skills, compute their current error with respect to the
        desired values, which is then returned in the vector svec. If the skills are encoded in task space, but the
        robot is controlled in joint space, the Jacobian matrix corresponding to the skills is also returned as smat.

        Parameters
        ----------
        :param x: current skills' control values
        :param desired_: desired skills' control values (as shown during training)

        Returns
        -------
        :return smat: skills Jacobian (stacked into one matrix)
        :return svec: vector of skills errors (between current and desired control values)
        """
        x = x.cpu().numpy()
        self.skill.update_desired_value(desired_.cpu().numpy(), use_state_idx=True)

        if self.is_train:
            use_state_idx = True
        else:
            use_state_idx = False

        if self.use_state_space:
            svec = self.skill.error(x, use_state_idx)
        else:
            svec = self.skill.error_from_fk(x)

        svec = torch.from_numpy(svec)
        svec = torch.unsqueeze(svec, -1)

        if self.use_state_space:
            return None, svec
        else:
            smat = self.skill.jacobian(x)
            smat = torch.from_numpy(smat)
            return smat, svec

    def get_extended_weight_vector(self, skill_weights):
        """
        This function extends a weight vector aimed to fill in the diagonal part of the weight matrix, so that the
        weight applies to every control value of the skill. The skills weights are thus duplicated for each control
        value.
        For instance, considering two skills of 2 and 3-dimensional control values, the function receives a vector
        (w1, w2) and returns a vector (w1, w1, w2, w2, w2).

        Parameters
        ----------
        :param skill_weights: vector of weights for the individual skills (one weight per skill)

        Returns
        -------
        :return: vector with duplicated weights in function of the number of control values for each skill
        """
        if len(skill_weights.shape) < 2:
            skill_weights = torch.unsqueeze(skill_weights, 0)

        ws = []
        for i in range(self.skill.n_skills):
            for j in range(self.skill.skills[i].dim()):
                ws.append(skill_weights[:, i])

        return torch.stack(ws, dim=1)

    def save_policy(self, model_fpath):
        torch.save(self.state_dict(), model_fpath)

    def load_policy(self, model_fpath):
        self.load_state_dict(torch.load(model_fpath))

    def forward(self, features, q, desired_):
        """
        This function computes the desired joint velocity for a given input.
        The parameters of the weight matrix are computed by one or some neural network layers and then serve as input
        for the parameters of a QP problem. An OptNet layer is used to solve the QP problem and obtain the desired joint
        velocities.
        This function needs to be implemented for each SkillPolicy class.

        Parameters
        ----------
        :param features: input features
        :param q: current joint position
        :param desired_: desired skill values

        Returns
        -------
        :return
        """
        raise NotImplementedError


class SkillDiagonalWeightedPolicy(SkillPolicy):
    """
    This SkillPolicy-based class sequences and blends skills using a diagonal weight matrix Q as input for the
    differentiable-optimization layer Optnet.
    """

    def __init__(self, dim, fdim, skill, eqn=None, ineqn=None, constant_skill_weights=None, eps=1e-10, dt=0.01):
        """
        Initialization of the SkillDiagonalWeightedPolicy class.

        Parameters
        ----------
        :param dim: TODO can we remove this parameter?
        :param fdim: the dimension of the features for the parameters of the QP
        :param skill: a list of skills

        Optional parameters
        -------------------
        :param eqn: equality constraints of the form Ax = b
        :param ineqn: inequality constraints of the form Gx < h
        :param eps: regularization value
        :param dt: time step
        """
        super(SkillDiagonalWeightedPolicy, self).__init__(dim, fdim, skill, eqn, ineqn, constant_skill_weights, eps, dt)
        # FullyConnectedNet takes the features as input and outputs a weight vector indicating the weight of each skill
        self.Qnet = FullyConnectedNet(fdim, self.skill.n_skills)

    def forward(self, features, q, desired_):
        """
        This function computes the desired joint velocity for a given input.
        The parameters of the weight matrix are computed by a fully-connected layer and then serve as input for the
        parameters of a QP problem. An OptNet layer is used to solve the QP problem and obtain the desired joint
        velocities.
        (Note: if we need some extra constraints on the weights vector, we can add them into the main training loop.)

        Parameters
        ----------
        :param features: input features
        :param q: current joint position
        :param desired_: desired skill values

        Returns
        -------
        :return dx: desired joint velocity
        :return wmat: weights matrix
        :return l_data: diagonal part of the weight matrix
        """
        # Calculate the weight matrix (diagonal)
        l_data = self.Qnet(features)
        if len(l_data.shape) < 2:
            l_data = torch.unsqueeze(l_data, dim=0)

        if self.skill.skill_cluster_idx is not None:
            d_datas = []
            for i in range(len(self.skill.skill_cluster_idx) - 1):
                d_datas.append(
                    nn.Softmax(dim=-1)(l_data[:, self.skill.skill_cluster_idx[i]:self.skill.skill_cluster_idx[i + 1]]))

            d_datas.append(nn.Softmax(dim=-1)(l_data[:, self.skill.skill_cluster_idx[-1]:]))
            l_data = torch.cat(d_datas, dim=-1)
        else:
            l_data = nn.Softmax(dim=-1)(l_data)

        l_data = self.get_extended_weight_vector(l_data)
        wmat = fill_diag(l_data)

        # Calculate current skill matrices
        wmat = torch.squeeze(wmat)
        if wmat.ndim == 2:
            wmat = wmat.unsqueeze(0)
        wmat = wmat.to(device)
        smat, svec = self.get_skill_mat(q, desired_)
        svec = svec.to(device)

        # Add artificial weighting of skills (to account for important differences in magnitude)
        if self.constant_skill_weights is not None:
            self.constant_skill_weights = self.constant_skill_weights.to(device)
            constant_skill_weights = self.get_extended_weight_vector(self.constant_skill_weights)
            constant_skill_weights_matrix = torch.squeeze(fill_diag(constant_skill_weights))
            constant_skill_weights_matrix = torch.unsqueeze(constant_skill_weights_matrix, dim=0)
            constant_skill_weights_matrix = torch.repeat_interleave(constant_skill_weights_matrix,
                                                                    repeats=wmat.shape[0], dim=0)
            updated_wmat = torch.matmul(torch.matmul(torch.transpose(constant_skill_weights_matrix, -1, -2), wmat),
                                        constant_skill_weights_matrix)
        else:
            updated_wmat = wmat

        # Calculate Q (the quadratic matrix)
        if smat is None:
            Q = updated_wmat
            p = - torch.matmul(torch.transpose(svec, -1, -2), updated_wmat)
        else:
            smat = smat.to(device)
            Q = torch.matmul(torch.matmul(torch.transpose(smat, -1, -2), updated_wmat), smat)
            p = - torch.matmul(torch.matmul(torch.transpose(svec, -1, -2), updated_wmat), smat)

        Q = Q + self.eps * torch.eye(Q.shape[1]).to(device)

        # Calculate the linear part of the quadratic cost
        p = torch.squeeze(p) / self.dt
        if p.ndim == 1:
            p = p.unsqueeze(0)

        # Solve the quadratic programming
        dx = QPFunction(verbose=-1, maxIter=1000)(Q, p, self.ineqn.G, self.ineqn.h, self.eqn.A, self.eqn.b)
        self.Q = Q
        self.wmat = wmat

        # dx = torch.where(torch.isnan(dx), 0.0, dx)
        return dx, wmat, l_data


class SkillSPDPolicy(SkillPolicy):
    """
    This SkillPolicy-based class sequences and blends skills is a basis class for policies using complete SPD weight
    matrix Q as input for the differentiable-optimization layer Optnet. Child classes need to specify the
    create_spd_matrix method.
    This class serves as basis for the SkillFullyWeightedPolicy.
    """
    def __init__(self, dim, fdim, skill, eqn=None, ineqn=None, constant_skill_weights=None, eps=1e-10, dt=0.01):
        """
        Initialization of the SkillSPDPolicy class.

        Parameters
        ----------
        :param fdim: the dimension of the features for the parameters of the QP
        :param skills: a list of skills

        Optional parameters
        -------------------
        :param eqn: equality constraints of the form Ax = b
        :param ineqn: inequality constraints of the form Gx < h
        :param eps: regularization value
        :param dt: time step
        """
        super(SkillSPDPolicy, self).__init__(dim, fdim, skill, eqn, ineqn, constant_skill_weights, eps, dt)
        self.QDiagNet = None
        self.QOffDiagNet = None

    def create_spd_matrix(self, wmat, l_data):
        raise NotImplementedError

    def forward(self, features, x, desired_):
        """
        This function computes the desired joint velocity for a given input.
        The parameters of the weight matrix are computed based on two networks (one for the diagonal values, another one
        for the off-diagonal elements) and then serve as input for the parameters of a QP problem. An OptNet layer is
        used to solve the QP problem and obtain the desired joint velocities.
        (Note: if we need some extra constraints on the weights vector, we can add them into the main training loop.)

        Parameters
        ----------
        :param features: input features
        :param q: current joint position
        :param desired_: desired skill values

        Returns
        -------
        :return dx: desired joint velocity
        :return wmat: weights matrix
        :return d_data: diagonal part of the weight matrix
        """
        if self.QDiagNet is None:
            raise ValueError('QDiagNet should not be None')
        if self.QOffDiagNet is None:
            raise ValueError('QOffDiagNet should not be None')

        # Calculate the weight matrix (full)
        d_data = self.QDiagNet(features)
        l_data = self.QOffDiagNet(features)
        if len(d_data.shape) < 2:
            d_data = torch.unsqueeze(d_data, dim=0)

        if self.skill.skill_cluster_idx is not None:
            d_datas = []
            for i in range(len(self.skill.skill_cluster_idx) - 1):
                d_datas.append(
                    nn.Softmax(dim=-1)(d_data[:, self.skill.skill_cluster_idx[i]:self.skill.skill_cluster_idx[i + 1]]))

            d_datas.append(nn.Softmax(dim=-1)(d_data[:, self.skill.skill_cluster_idx[-1]:]))
            d_data = torch.cat(d_datas, dim=-1)
        else:
            d_data = nn.Softmax(dim=-1)(d_data)

        # Calculate current skill matrices
        d_data = self.get_extended_weight_vector(d_data)
        wmat = fill_diag(d_data)
        wmat = self.create_spd_matrix(wmat, l_data)
        wmat = wmat.to(device)

        # Add artificial weighting of skills (to account for important differences in magnitude)
        if self.constant_skill_weights is not None:
            self.constant_skill_weights = self.constant_skill_weights.to(device)
            constant_skill_weights_matrix = fill_diag(self.constant_skill_weights)
            updated_wmat = torch.matmul(torch.matmul(torch.transpose(constant_skill_weights_matrix, -1, -2), wmat),
                                        constant_skill_weights_matrix)
        else:
            updated_wmat = wmat

        smat, svec = self.get_skill_mat(x, desired_)
        if smat is not None:
            smat = smat.to(device)

        svec = svec.to(device)
        if smat is not None:
            Q = torch.matmul(torch.matmul(torch.transpose(smat, -1, -2), updated_wmat), smat)
            p = - torch.matmul(torch.matmul(torch.transpose(svec, -1, -2), updated_wmat), smat)
        else:
            Q = updated_wmat
            p = - torch.matmul(torch.transpose(svec, -1, -2), updated_wmat)

        Q = Q + self.eps * torch.eye(Q.shape[1]).to(device)

        # Calculate the linear part of the quadratic cost
        p = torch.squeeze(p) / self.dt

        # Solve the quadratic programming
        dx = QPFunction(verbose=-1, maxIter=1000)(Q, p, self.ineqn.G, self.ineqn.h, self.eqn.A, self.eqn.b)
        self.Q = Q
        self.wmat = wmat
        return dx, wmat, d_data


class SkillFullyWeightedPolicy(SkillSPDPolicy):
    """
    This SkillPolicy-based class sequences and blends skills using a complete SPD weight matrix Q as input for the
    differentiable-optimization layer Optnet.
    """
    def __init__(self, dim, fdim, skill, eqn=None, ineqn=None, constant_skill_weights=None, eps=1e-10, dt=0.01):
        """
        Initialization of the SkillFullyWeightedPolicy class.

        Parameters
        ----------
        :param fdim: the dimension of the featuresures for the parameters of the QP
        :param skills: a list of skills

        Optional parameters
        -------------------
        :param eqn: equality constraints of the form Ax = b
        :param ineqn: inequality constraints of the form Gx < h
        :param eps: regularization value
        :param dt: time step
        """
        super(SkillFullyWeightedPolicy, self).__init__(dim, fdim, skill, eqn, ineqn, constant_skill_weights, eps, dt)
        n_skills = self.skill.n_skills
        total_dim = self.skill.total_dim
        self.QDiagNet = FullyConnectedNet(fdim, n_skills)
        nb_offdiag_parameters = int(0.5 * (total_dim ** 2 - np.sum(np.array(self.skill.skills_dim) ** 2)) + n_skills - 1)
        self.QOffDiagNet = FullyConnectedNet(fdim, nb_offdiag_parameters)

    def create_spd_matrix(self, wmat, l_data):
        """
        This function builds a SPD matrix based on diagonal elements and contraction matrices in a recursive manner.
        Namely, given diagonal weights w_1 and w_2 for the first two skills and a contraction matrix K12, the
        off-diagonal elements are computed as W12 = msqrt(Y) K12 msqrt(Z) with Y = w_1 I, Z = w_2 I. This is then done
        recursively, such that (W13' W23')' =  msqrt(Y) K3 msqrt(Z) with Y = (w1 I W12; W12' w2 I), z = w3 I.
        The contraction matrices have a norm <=1 and are computed as v U / ||U||, where w and U are obtained from a
        fully-connected layer.

        Parameters
        ----------
        :param wmat: diagonal weight matrix (off-diagonal elements are 0)
        :param l_data: buiding elements for the contraction matrices (all v come first, all U follow)

        Returns
        -------
        :return wmat: full SPD weight matrix
        """
        if len(l_data.shape) < 2:
            l_data = torch.unsqueeze(l_data, 0)
        l_data[:, :self.skill.n_skills - 1] = nn.Sigmoid()(l_data[:, :self.skill.n_skills - 1])
        l_data[:, self.skill.n_skills - 1:] = nn.Tanh()(l_data[:, self.skill.n_skills - 1:])

        cumsum_dim = np.array([0] + self.skill.skills_dim).cumsum()
        start_param_id = self.skill.n_skills - 1
        end_param_id = start_param_id

        for i in range(1, self.skill.n_skills):
            end_param_id = end_param_id + cumsum_dim[i] * self.skill.skills_dim[i]
            norm_contraction_matrix = l_data[:, i - 1]
            contraction_matrix = l_data[:, start_param_id:end_param_id].reshape(-1, cumsum_dim[i], self.skill.skills_dim[i])
            contraction_matrix = contraction_matrix / torch.linalg.norm(contraction_matrix, dim=(-2, -1))[:, None, None]
            contraction_matrix = contraction_matrix * norm_contraction_matrix[:, None, None]
            left_top_matrix = wmat[:, :cumsum_dim[i], :cumsum_dim[i]].detach()
            right_bottom_matrix = wmat[:, cumsum_dim[i]:cumsum_dim[i + 1], cumsum_dim[i]:cumsum_dim[i + 1]].detach()
            sqrt_left_top_matrix = sqrtm_torch(left_top_matrix)
            sqrt_right_bottom_matrix = torch.sqrt(right_bottom_matrix)  # assume a diagonal matrix here
            off_diagonal_matrix = torch.matmul(torch.matmul(sqrt_left_top_matrix, contraction_matrix),
                                               sqrt_right_bottom_matrix)
            wmat[:, 0:cumsum_dim[i], cumsum_dim[i]:cumsum_dim[i] + self.skill.skills_dim[i]] = off_diagonal_matrix
            wmat[:, cumsum_dim[i]:cumsum_dim[i] + self.skill.skills_dim[i],
            0:cumsum_dim[i]] = off_diagonal_matrix.transpose(-2, -1)
            start_param_id = end_param_id

        wmat = torch.squeeze(wmat)
        return wmat

    def spec_train(self, dataloader, model_path, loss_weights=None, learning_rate=0.001, max_epochs=500,
                   proportional=0.2):
        """
        This function trains a fully-weighted policy in a hierarchical manner. First, the fully connected layer
        corresponding to the diagonal of the weight matrix is trained (similarly as a SkillDiagonalPolicy class). The
        obtained network is then used as initialization for the training of the two networks encoding the full SPD
        weight matrix.

        Parameters
        ----------
        :param dataloader: dataloader containing the training data
        :param model_path: path where to save the policy

        Optional parameters
        -------------------
        :param loss_weights: artificial weighting of skills (to account for important differences in magnitude)
        :param learning_rate: learning rate for the networks training
        :param max_epochs: maximum number of epochs for the network training
        :param proportional: sets the number of pre-training epochs (for the diagonal part) as proportional*max_epochs

        Returns
        -------
        :return: trained policy
        """

        policy = SkillDiagonalWeightedPolicy(self.dim, self.fdim, self.skill, eqn=self.eqn, ineqn=self.ineqn)
        print('Pretrain: training SkillDiagonalWeightedPolicy')
        pretrain_epochs = int(proportional * max_epochs)
        policy = train_policy(policy, dataloader, model_path, loss_weights=loss_weights,
                              learning_rate=learning_rate, max_epochs=pretrain_epochs,
                              consider_spec_train=False)
        self.QDiagNet = policy.Qnet

        if max_epochs - pretrain_epochs == 0:
            return self
        else:
            return train_policy(self, dataloader, model_path, loss_weights=loss_weights,
                                learning_rate=learning_rate, max_epochs=max_epochs-pretrain_epochs,
                                consider_spec_train=False)


def train_policy(policy, dataloader, model_path, spec_train_proportional=0.2, loss_weights=None, learning_rate=0.001,
                 max_epochs=500, consider_spec_train=True):
    """
    This function trains the parameters of a SkillPolicy. Namely, it trains the parameters of the network(s) used to
    compute the weight matrix, which are then given as inputs to the Optnet layer.

    Parameters
    ----------
    :param dataloader: dataloader containing the training data
    :param model_path: path where to save the policy
    :param loss_weights: artificial weighting of skills (to account for important differences in magnitude)
    :param learning_rate: learning rate for the networks training
    :param max_epochs: maximum number of epochs for the network training
    :param spec_train_proportional: sets the number of pre-training epochs (for the diagonal part) as proportional*max_epochs
    :param consider_spec_train: if True, trains a fully matrix in two stages, by pre-training the diagonal part

    Returns
    -------
    :return: trained policy
    """

    if consider_spec_train and hasattr(policy, 'spec_train') and callable(policy.spec_train):
        return policy.spec_train(dataloader, model_path, loss_weights, learning_rate, max_epochs,
                                 spec_train_proportional)

    opt = optim.Adam(policy.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10000, verbose=False, min_lr=1e-5)
    loss_weights_ = [[loss_weights[i]] * policy.skill.skills[i].dim() for i in range(len(policy.skill.skills))]
    loss_weights_ = [item for sublist in loss_weights_ for item in sublist]
    loss_weights_ = torch.tensor(loss_weights_, dtype=torch.float64)
    loss_weights_ = loss_weights_.to(device)

    qp_loss = MultipleSkillsVelocityKktLoss()

    for epoch in range(max_epochs):
        eloss = 0
        count = 0
        
        # import time
        # start = time.time()
        for i, (feat, xt, dxt, desired_) in enumerate(dataloader):
            feat = feat.squeeze(dim=0)
            xt = xt.squeeze(dim=0)
            dxt = dxt.squeeze(dim=0)
            desired_ = desired_.squeeze(dim=0)

            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            if xt.ndim == 1:
                xt = xt.unsqueeze(0)
            if dxt.ndim == 1:
                dxt = dxt.unsqueeze(0)
            if desired_.ndim == 1:
                desired_ = desired_.unsqueeze(0)

            opt.zero_grad()
            if hasattr(policy, "ineqn") and policy.ineqn is not None:
                policy.ineqn.update(xt)

            if hasattr(policy, "eqn") and policy.eqn is not None:
                policy.eqn.update(xt)

            feat = feat.to(device)
            xt = xt.to(device)
            dxt = dxt.to(device)
            desired_ = desired_.to(device)
            dxt_, wmat, ldata = policy.forward(feat, xt, desired_)
            
            loss = qp_loss(dxt_, dxt, wmat, loss_weights_)

            loss.backward()
            opt.step()
            eloss += loss
            count += 1

        if epoch % 20 == 0:
            policy.save_policy(model_path)
            # torch.save(policy, model_path)

        cost = eloss / count
        lr_scheduler.step(cost)
        print('epoch: %1d / %1d, loss: %.10f, lr: %.10f' % (epoch, max_epochs, cost, opt.param_groups[0]['lr']), end='\r')

    print('epoch: %1d, loss: %.10f,  lr: %.10f' % (epoch, cost, opt.param_groups[0]['lr']))
    return policy


