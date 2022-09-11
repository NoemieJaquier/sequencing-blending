import numpy as np
import roboticstoolbox as rtb

from SkillsSequencing.utils.robot_plot_utils import plot_planar_robot
from SkillsSequencing.utils.orientation_utils import rotation_matrix_to_unit_sphere


class Gripper2D:
    def __init__(self, nb_dofs_arm=4, nb_dofs_finger=3,  arm_link_length=4, gripper_link_length=2):
        """
        Initialize a 2D robot with gripper using the robotics toolbox.

        Optional parameters
        -------------------
        :param nb_dofs_arm: number of DoF for the arm
        :param nb_dofs_finger: number of DoF for each of the two fingers of the gripper
        :param arm_link_length: length of the links of the arm
        :param gripper_link_length: length of the links of the fingers
        """
        link_arm = rtb.RevoluteDH(d=0., a=arm_link_length, alpha=0.)
        link_finger = rtb.RevoluteDH(d=0., a=gripper_link_length, alpha=0.)
        links_list = [link_arm for _ in range(nb_dofs_arm)] + [link_finger for _ in range(nb_dofs_finger)]

        # Robot
        self.robot = rtb.DHRobot(links_list)
        # Link lengths (arm and fingers)
        self.link_lengths = np.array([arm_link_length] * nb_dofs_arm + [gripper_link_length] * nb_dofs_finger)
        # Total number of DoFs (arm + 2 x fingers)
        self.total_links = nb_dofs_arm + 2 * nb_dofs_finger
        # Ids of arm joints
        self.arm_idx = [i for i in range(nb_dofs_arm)]
        # Ids of left and right finger of the gripper
        self.lgripper_idx = [i + nb_dofs_arm for i in range(nb_dofs_finger)]
        self.rgripper_idx = [i + nb_dofs_arm + nb_dofs_finger for i in range(nb_dofs_finger)]
        # Ids of the gripper (left + right finger)
        self.gripper_idx = self.lgripper_idx + self.rgripper_idx
        # Ids for the kinematic chain of the arm and left/right gripper finger
        self.lf_idx = self.arm_idx + [i + nb_dofs_arm for i in range(nb_dofs_finger)]
        self.rf_idx = self.arm_idx + [i + nb_dofs_arm + nb_dofs_finger for i in range(nb_dofs_finger)]

    def plot(self, ax, q, facecolor):
        """
        Plot the 2D arm at the given joint position.

        Parameters
        ----------
        :param q: current joint angles      (nb_joint)
        :param facecolor: color of the robot for the plot

        Return
        ------
        :return: -
        """
        patch_list_left = plot_planar_robot(ax, q[self.lf_idx], self.link_lengths, facecolor=facecolor)
        patch_list_right = plot_planar_robot(ax, q[self.rf_idx], self.link_lengths, facecolor=facecolor)
        return patch_list_left + patch_list_right

    def arm_position_fct(self, q):
        """
        Computes the arm end-effector position using forward kinematics. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: end-effector position      (nb_data x 3)
        """
        batch_size = np.shape(q)[0]
        while q.shape[-1] < self.total_links:
            q = np.concatenate([q, [[0.0]]], axis=-1)

        fct = [self.robot.fkine_all(q[i, self.lf_idx]).t[self.arm_idx[-1]+1, 0:2] for i in range(batch_size)]
        return np.stack(fct)

    def arm_position_jacobian_fct(self, q):
        """
        Computes the position part of the arm Jacobian given the current joint angle. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: position part of the Jacobian (nb_data x 3 x nb_joints)
        """
        batch_size = q.shape[0]
        while q.shape[-1] < self.total_links:
            q = np.concatenate([q, [[0.0]]], axis=-1)

        jacob = [self.robot.jacob0(q[i, self.lf_idx])[0:2, self.arm_idx] for i in range(batch_size)]
        return np.stack(jacob)

    def arm_orientation_fct(self, q):
        """
        Computes the arm end-effector orientation using forward kinematics. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: end-effector orientation   (nb_data x 4)
        """
        batch_size = np.shape(q)[0]
        while q.shape[-1] < self.total_links:
            q = np.concatenate([q, [[0.0]]], axis=-1)

        fct = [rotation_matrix_to_unit_sphere(self.robot.fkine_all(q[i, self.lf_idx]).R[self.arm_idx[-1]+1, 0:2, 0:2])
               for i in range(batch_size)]
        return np.stack(fct)

    def arm_orientation_jacobian_fct(self, q):
        """
        Computes the orientation part of the arm Jacobian given the current joint angle. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: orientation part of the Jacobian (nb_data x 3 x nb_joints)
        """
        batch_size = q.shape[0]
        while q.shape[-1] < self.total_links:
            q = np.concatenate([q, [[0.0]]], axis=-1)

        jacob = [self.robot.jacob0(q[i, self.lf_idx])[-1, self.arm_idx][None] for i in range(batch_size)]
        return np.stack(jacob)

    def compute_ts_arm_fct(self, q):
        """
        Computes the arm end-effector pose using forward kinematics. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: end-effector pose          (nb_data x 7)
        """
        pos_fct = self.arm_position_fct(q)
        ori_fct = self.arm_orientation_fct(q)
        x_fct = np.concatenate([pos_fct, ori_fct], axis=-1)
        return x_fct

    def compute_ts_jacobian_arm_fct(self, q):
        """
        Computes the arm Jacobian given the current joint angle. Batch computation is supported.

        Parameters
        ----------
        :param q: current joint angles      (nb_data x nb_joint)

        Return
        ------
        :return: Jacobian                   (nb_data x 6 x nb_joints)
        """
        pos_jacob = self.arm_position_jacobian_fct(q)
        ori_jacob = self.arm_orientation_jacobian_fct(q)
        x_jacob = np.concatenate([pos_jacob, ori_jacob], axis=1)
        return x_jacob

    def close_skill(self, t):
        """
        Returns the desired position for the close-gripper skill
        """
        return np.array([np.pi/2, -np.pi/2, -np.pi/3, -np.pi/2, np.pi/2, np.pi/3])

    def open_skill(self, t):
        """
        Returns the desired position for the open-gripper skill
        """
        return np.array([np.pi/2, -np.pi/3, 0, -np.pi/2, np.pi/3, 0])

    def stop_arm_skill(self, t):
        """
        Returns the desired position for the stop-arm skill
        """
        return np.array([0, 0, 0, 0])

    def stop_gripper_skill(self, t):
        """
        Returns the desired position for the stop-gripper skill
        """
        return np.array([0, 0, 0, 0, 0, 0])

