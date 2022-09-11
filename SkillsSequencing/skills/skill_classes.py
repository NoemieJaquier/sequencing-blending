import abc
import numpy as np
from SkillsSequencing.utils.orientation_utils import sphere_logarithmic_map_batch
from SkillsSequencing.utils.orientation_utils import compute_analytical_orientation_jacobian_sphere_batch
from SkillsSequencing.utils.matrices_processing import assign_matrix


class Skill(object):
    """
    Abstract class setting out a template for skills classes. Skills should inherit from this class.
    """
    def __init__(self, desired_value):
        """
        This function initializes the skill class.

        Parameters
        ----------
        :param desired_value: desired value for the skill
        """
        self.desired_value = desired_value
        super().__init__()

    @abc.abstractmethod
    def dim(self):
        """
        This function computes the dimension of the skill in vector form (equal to the dimension of the error vector).

        Returns
        -------
        :return: dimension of the skill
        """

    @abc.abstractmethod
    def error(self, x):
        """
        This function computes the error of the current value of the skill with respect to the desired value.

        Parameters
        ----------
        :param x: current skill value

        Returns
        -------
        :return:
        """

    @abc.abstractmethod
    def error_from_fk(self, q):
        """
        This function computes the error of the current value of the skill with respect to the desired value.
        The current skill value is computed from joint angles values with forward kinematics

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return:
        """

    @abc.abstractmethod
    def jacobian(self, q):
        """
        This function computes the Jacobian J of the skill such that dx = Jdq, with dx the velocity (change of) the
        skill x and dq the joint velocity

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: Jacobian matrix
        """

    def update_desired_value(self, desired_value):
        """
        This function updates the desired value of the skill saved in the class.

        Parameters
        ----------
        :param desired_value: desired value for the skill

        Returns
        -------
        :return: -
        """
        self.desired_value = desired_value


class JointStopSkillBatch(Skill):
    """
    Instances of this class are skills related to the joint angles (q) of the robot.
    """

    def __init__(self, desired_value, name=None, config_idx=None, state_idx=None):
        super().__init__(desired_value)
        self._dim = desired_value.shape[-1]
        self.config_idx = config_idx
        self.state_idx = state_idx
        if name is None:
            self.name = type(self).__name__ + "_" + np.random.randint(0,10)
        else:
            self.name = name

    def dim(self):
        return self._dim

    def update_desired_value(self, desired_value, use_state_idx=False):
        if len(desired_value.shape) < 2:
            desired_value = np.expand_dims(desired_value, axis=0)

        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def error(self, dq):
        if dq.shape[-1] > self.desired_value.shape[-1]:
            if self.state_idx is not None:
                dq = dq[:, self.state_idx]
            else:
                dq = dq[:, :self.desired_value.shape[-1]]

        if dq.shape[0] != self.desired_value.shape[0]:
            raise ValueError('q should have the same batch size as the desired value')

        return np.zeros(shape=dq.shape)

    def error_from_fk(self, dq):
        if self.config_idx is not None:
            dq = dq[:, self.config_idx]

        return self.error(dq)

    def jacobian(self, dq):
        if self.config_idx is not None:
            dq = dq[:, self.config_idx]

        jac = np.eye(dq.shape[-1])
        jac = np.expand_dims(jac, axis=0)
        jac = np.tile(jac, (dq.shape[0], 1, 1))
        return jac


class JointPositionSkillBatch(Skill):
    """
    Instances of this class are skills related to the joint angles (q) of the robot.
    """

    def __init__(self, desired_value, name=None, config_idx=None, state_idx=None):
        super().__init__(desired_value)
        self._dim = desired_value.shape[-1]
        self.config_idx = config_idx
        self.state_idx = state_idx
        self.orignal_state_idx = state_idx
        if name is None:
            self.name = type(self).__name__ + "_" + str(np.random.randint(0,10))
        else:
            self.name = name

    def reset_state_idx(self):
        self.state_idx = self.orignal_state_idx

    def update_desired_value(self, desired_value, use_state_idx=False):
        if len(desired_value.shape) < 2:
            desired_value = np.expand_dims(desired_value, axis=0)

        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def dim(self):
        return self._dim

    def error(self, q, use_state_idx=False):
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)

        if q.shape[-1] > self.desired_value.shape[-1]:
            if self.state_idx is not None and use_state_idx:
                q = q[:, self.state_idx]
            else:
                q = q[:, self.orignal_state_idx]
                
        if q.shape[0] != self.desired_value.shape[0]:
            raise ValueError('q should have the same batch size as the desired value')

        return self.desired_value - q

    def error_from_fk(self, q):
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        return self.error(q)

    def jacobian(self, q):
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        jac = np.eye(q.shape[-1])
        jac = np.expand_dims(jac, axis=0)
        jac = np.tile(jac, (q.shape[0],1,1))
        return jac


class TaskspaceSkillBatch(Skill):
    def __init__(self, desired_value, pos_dim=2, name=None, compute_fct_batch=None,
                 compute_jacobian_fct_batch=None, config_idx=None, state_idx=None):
        super().__init__(desired_value)
        self.compute_ts_fct_batch = compute_fct_batch
        self.compute_ts_jacobian_fct_batch = compute_jacobian_fct_batch
        self.dim_ = desired_value.shape[-1]
        self.config_idx = config_idx
        self.state_idx = state_idx
        self.orignal_state_idx = state_idx
        if name is None:
            self.name = type(self).__name__ + "_" + str(np.random.randint(0,10))
        else:
            self.name = name

        self.pos_dim = pos_dim

    def reset_state_idx(self):
        self.state_idx = self.orignal_state_idx

    def update_desired_value(self, desired_value, use_state_idx=False):
        if len(desired_value.shape) < 2:
            desired_value = np.expand_dims(desired_value, axis=0)
        
        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def dim(self):
        return self.dim_

    def error(self, x, use_state_idx=False):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[-1] > self.desired_value.shape[-1]:
            if self.state_idx is not None and use_state_idx:
                x = x[:, self.state_idx]
            else:
                x = x[:, self.orignal_state_idx]

        tx = x[:, :self.pos_dim]
        terror = self.desired_value[:, :self.pos_dim] - tx
        rx = x[:, self.pos_dim:]

        rerror = sphere_logarithmic_map_batch(self.desired_value[:, self.pos_dim:], rx)
        return np.concatenate([terror, rerror], axis=-1)

    def error_from_fk(self, q):
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)

        if self.config_idx is not None:
            q = q[:, self.config_idx]
        
        x = self.compute_ts_fct_batch(q)
        return self.error(self.compute_ts_fct_batch(q))

    def jacobian(self, q):
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        ts_fct = self.compute_ts_fct_batch(q)
        ts_jacobian = self.compute_ts_jacobian_fct_batch(q)

        if len(ts_jacobian.shape) < 3:
            ts_jacobian = np.expand_dims(ts_jacobian, axis=0)

        if len(ts_fct.shape) < 2:
            ts_fct = np.expand_dims(ts_fct, axis=0)

        if self.pos_dim == 2:
            pos_jac = ts_jacobian[:, :-1,:]
            current_orientation = ts_fct[:, self.pos_dim:]
            orientation_jacobian = ts_jacobian[:, -1, :][:, np.newaxis,:]
            ori_jac = compute_analytical_orientation_jacobian_sphere_batch(current_orientation, orientation_jacobian)
            jac = np.concatenate([pos_jac, ori_jac], axis=1)
        else:
            pos_jac = ts_jacobian[:, :self.pos_dim, :]
            current_orientation = ts_fct[:, self.pos_dim:]

            orientation_jacobian = ts_jacobian[:, self.pos_dim:, :][:, np.newaxis, :]
            ori_jac = compute_analytical_orientation_jacobian_sphere_batch(current_orientation, orientation_jacobian, self.pos_dim)
            jac = np.concatenate([pos_jac, ori_jac], axis=1)

        return jac


class PositionSkillBatch(Skill):
    def __init__(self, desired_value, name=None,
                 compute_fct_batch=None,
                 compute_jacobian_fct_batch=None, config_idx=None, state_idx=None):
        """
        The batch version of PositionSkill (see skill_classes). The functions should support batch calculation including
        batch matrix operation.
        """
        super(PositionSkillBatch, self).__init__(desired_value)
        self.compute_position_fct = compute_fct_batch
        self.compute_position_jacobian_fct = compute_jacobian_fct_batch
        self.desired_value = desired_value
        self.dim_ = desired_value.shape[-1]
        self.config_idx = config_idx
        self.state_idx = state_idx

        if name is None:
            self.name = type(self).__name__ + "_" + str(np.random.randint(0,10))
        else:
            self.name = name

    def update_desired_value(self, desired_value, use_state_idx=False):
        if len(desired_value.shape) < 2:
            desired_value = np.expand_dims(desired_value, axis=0)

        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def dim(self):
        """
        This function computes the dimension of the skill (equal to the dimension of the error vector).

        Returns
        -------
        :return: dimension of the skill
        """
        return self.dim_

    def error(self, x):
        """
        This function computes the error of the current end-effector position with respect to the desired one.

        Parameters
        ----------
        :param x: current end-effector position

        Returns
        -------
        :return: xpdes - xp
        """
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[-1] > self.desired_value.shape[-1]:
            if self.state_idx is not None:
                x = x[:, self.state_idx]
            else:
                x = x[:, :self.desired_value.shape[-1]]

        if x.shape[0] != self.desired_value.shape[0]:
            raise ValueError('x should have the same batch size as the desired value')

        return self.desired_value - x

    def error_from_fk(self, q):
        """
        This function computes the error of the current end-effector position with respect to the desired one.

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: xpdes - xp
        """
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        if q.shape[0] != self.desired_value.shape[0]:
            raise ValueError('q should have the same batch size as the desired value')

        current_position = self.compute_position_fct(q)
        return self.error(current_position)

    def error_gradient_from_fk(self, q):
        """
        This function computes the derivative of the error of the current end-effector position compared to the desired
        one with respect to joint position.

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: - dxp/dq
        """
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        return -self.compute_position_jacobian_fct(q)

    def jacobian(self, q):
        """
        This function computes the Jacobian J which relates the change in task space position to the change of joint
        angles, such that dx = Jdq.

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: Jacobian matrix
        """
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        pos_jac = self.compute_position_jacobian_fct(q)
        if len(pos_jac.shape) < 3:
            pos_jac = np.expand_dims(pos_jac, axis=0)

        return pos_jac


class OrientationSkillBatch(Skill):
    """
    The batch version of OrientationSkill (see skill_classes). The functions should support batch calculation including
    batch matrix operation.
    """
    def __init__(self, desired_value, name=None, compute_fct_batch=None,
                 compute_jacobian_fct_batch=None, config_idx=None, state_idx=None):
        super().__init__(desired_value)
        self.compute_orientation_fct = compute_fct_batch
        self.compute_orientation_jacobian_fct = compute_jacobian_fct_batch
        self.dim_ = desired_value.shape[-1]
        self.config_idx = config_idx
        self.state_idx = state_idx
        if name is None:
            self.name = type(self).__name__ + "_" + str(np.random.randint(0,10))
        else:
            self.name = name

    def update_desired_value(self, desired_value, use_state_idx=False):
        if len(desired_value.shape) < 2:
            desired_value = np.expand_dims(desired_value, axis=0)

        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def dim(self):
        """
        This function computes the dimension of the skill (equal to the dimension of the error vector).

        Returns
        -------
        :return: dimension of the skill
        """
        return self.dim_

    def error(self, x):
        """
        This function computes the error of the current end-effector orientation with respect to the desired one.

        Parameters
        ----------
        :param x: current end-effector orientation

        Returns
        -------
        :return: Log_xo(desxo)
        """

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[-1] > self.desired_value.shape[-1]:
            if self.state_idx is not None:
                x = x[:, self.state_idx]
            else:
                x = x[:, :self.desired_value.shape[-1]]

        if x.shape[0] != self.desired_value.shape[0]:
            raise ValueError('x should have the same batch size as the desired value')

        return sphere_logarithmic_map_batch(self.desired_value, x)

    def error_from_fk(self, q):
        """
        This function computes the error of the current end-effector orientation with respect to the desired one.

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: Log_xo(desxo)
        """
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        if q.shape[0] != self.desired_value.shape[0]:
            raise ValueError('q should have the same batch size as the desired value')

        current_orientation = self.compute_orientation_fct(q)
        return self.error(current_orientation)

    def jacobian(self, q):
        """
        This function computes the Jacobian J which relates the change in task space orientation to the change of joint
        angles, such that dxo = Jdq.
        Note that, as the orientation is expressed with quaternions, the analytical Jacobian is considered.

        Parameters
        ----------
        :param q: joint angles

        Returns
        -------
        :return: Jacobian matrix
        """
        if self.config_idx is not None:
            q = q[:, self.config_idx]

        current_orientation = self.compute_orientation_fct(q)
        orientation_jacobian = self.compute_orientation_jacobian_fct(q)

        ori_jac = compute_analytical_orientation_jacobian_sphere_batch(current_orientation, orientation_jacobian)
        if len(ori_jac.shape) < 3:
            ori_jac = np.expand_dims(ori_jac, axis=0)

        return ori_jac


class SkillComplex():
    """
    This class encodes a set of skills and allows the computation of the error and Jacobian including all skills in
    combined matrices.
    """
    def __init__(self, state_dim, config_dim, skills, skill_cluster_idx=None):
        """
        Initialization of the class
        :param state_dim: sum of dimensions of all skills
        :param config_dim: sum of dimensions of control variable (joint-space dimension)
        :param skills: list of skills
        """
        self.skill_cluster_idx = skill_cluster_idx
        self.state_dim = state_dim
        self.config_dim = config_dim
        self.skills = skills
        self.n_skills = len(skills)
        self.skills_dim = []
        total_dim = 0
        for i in range(len(skills)):
            self.skills_dim.append(skills[i].dim())
            total_dim += skills[i].dim()

        self.total_dim = total_dim

    def update_desired_value(self, desired_value, use_state_idx=False):
        [skill.update_desired_value(desired_value, use_state_idx) for skill in self.skills]

    def error(self, state, use_state_index=True):
        err = [skill.error(state, use_state_index) for skill in self.skills]
        err = np.concatenate(err, axis=-1)
        return err

    def error_from_fk(self, config):
        err = [skill.error_from_fk(config) for skill in self.skills]
        err = np.concatenate(err, axis=-1)
        return err

    def jacobian(self, config):
        """
        A mapping from derivative of config to derivative of state
        d_state = J * d_config
        :param config: current configuration of the robot (such as joint values that will be directly controlled)
        :return:
        """

        jac = np.zeros((config.shape[0], self.state_dim, self.config_dim))
        # old_state_idx = 0
        for i in range(len(self.skills)):
            skill = self.skills[i]
            sjac = skill.jacobian(config)
            if skill.state_idx is None:
                state_idx = list(range(skill.dim()))
            else:
                state_idx = skill.state_idx

            if skill.config_idx is None:
                config_idx = list(range(config.shape[-1]))
            else:
                config_idx = skill.config_idx

            # state_idx = [idx + old_state_idx for idx in state_idx]
            jac = assign_matrix(sjac, jac, state_idx, config_idx)
            # old_state_idx = max(state_idx)

        return jac

    def expand_state_to_skill_state(self, x):
        xres = []
        for i in range(len(self.skills)):
            skill = self.skills[i]
            if skill.state_idx is None:
                state_idx = list(range(skill.dim()))
            else:
                state_idx = skill.state_idx

            xres.append(x[:, state_idx])
        
        xres = np.concatenate(xres, axis=-1)
        return xres



