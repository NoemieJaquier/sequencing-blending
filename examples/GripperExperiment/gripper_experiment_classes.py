import os
import matplotlib.pyplot as plt
import matplotlib

from SkillsSequencing.utils.orientation_utils import sphere_logarithmic_map_batch
from SkillsSequencing.qpnet.constraints import *
from SkillsSequencing.utils.utils import prepare_torch

device = prepare_torch()

plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
matplotlib.rc_file_defaults()


class Gripper2DExp:
    """
    This class defines a pick and place experiment for a planar robot with a gripper.
    """
    def __init__(self, skill_list, skill_fcns, gripper):
        """
        Initialization of the experiment class.

        Parameters
        ----------
        :param skill_list: list of skills
        :param skill_fcns: skills functions returning the desired control values for the list of skills
        :param gripper: robot with gripper

        """
        self.skill_list = skill_list
        self.skill_fcns = skill_fcns
        self.gripper = gripper

    def reset_demo_log_after_ndofs_change(self, q0):
        """
        This function resets the log of the demonstration data.

        Parameters
        ----------
        :param q0: initial joint position

        Return
        ------
        :return: -
        """
        nb_dofs = self.gripper.total_links
        self.demo_log['qt'] = None
        self.demo_log['dqt'] = None
        self.demo_log['K'] = self.demo_log['K'][0, 0] * np.eye(nb_dofs)
        self.demo_log['q0'] = q0

    def get_taskspace_training_data(self, fname):
        """
        This function returns the training data for a demonstration trajectory. The training data consists of the list
        of skills, the position and velocity along the trajectory, the desired skill values along the trajectory, and
        the list of skills acting on the same control type (e.g., end-effector-pose-related skills, hand skills, ...).

        Parameters
        ----------
        :param fname: file containing demonstration trajectory

        Return
        ------
        :return skill_list: list of skills
        :return xt_d: positions of the different skills along the demonstration
        :return dxt_d: velocities of the different skills along the demonstration
        :return desired_: desired control values given by the skills along the trajectory
        :return ctrl_idx: vector of control ids, skills with same control type share a control id
        """
        demo_log = np.load(fname)
        self.demo_log = dict(demo_log)

        if self.demo_log is None:
            raise ValueError('Please generate demo first')

        traj = self.demo_log['xpt']
        for i in range(len(self.skill_list)):
            desired_value = np.stack([self.skill_fcns[i](traj[t, :2]) for t in np.arange(0, traj.shape[0])])
            self.skill_list[i].update_desired_value(desired_value)

        # Prepare training data
        qt_d = self.demo_log['qt']
        xpt_d = self.demo_log['xpt']
        xot_d = self.demo_log['xot']
        K = self.demo_log['K'][0,0]
        dt = self.demo_log['dt']

        dqt_d = (qt_d[1:, :] - qt_d[:-1, :]) / (K * dt)
        dxpt_d = (xpt_d[1:, :] - xpt_d[:-1, :]) / (K * dt)
        dxot_d = sphere_logarithmic_map_batch(xot_d[:-1, :], xot_d[1:, :]) / (K * dt)

        xt = []
        dxt = []
        desired_ = []
        old_idx = 0
        ctrl_idx = []
        for skill in self.skill_list:
            n_added = 0
            if skill.name in {'pick', 'place'}:
                cxt = np.concatenate([xpt_d, xot_d], axis=-1)
                xt.append(cxt)
                dxt.append(np.concatenate([dxpt_d, dxot_d], axis=-1))
                desired_.append(skill.desired_value)
                n_added = cxt.shape[-1]
                ctrl_idx.append(0)
            if skill.name in {'open', 'close'}:
                qtx = qt_d[:, 4:]
                qtx_d = dqt_d[:, 4:]
                xt.append(qtx)
                dxt.append(qtx_d)
                desired_.append(skill.desired_value)
                n_added = qtx.shape[-1]
                ctrl_idx.append(1)

            skill.state_idx = list(np.arange(n_added) + old_idx)
            old_idx += n_added

        desired_ = np.hstack(desired_)
        xt_d = np.hstack(xt)
        dxt_d = np.hstack(dxt)
        return self.skill_list, xt_d, dxt_d, desired_, ctrl_idx

    def test_policy(self, policy, pick_goal, place_goal, is_plot=False, fname=None, is_plot_robot=False,
                    joint_position_limits=None, joint_velocity_limits=None):
        """
        Test a given policy. The robot is controlled in joint space.

        Parameters
        ---------
        :param policy: trained policy
        :param pick_goal: picking location
        :param place_goal: placing location

        Optional parameters
        -------------------
        :param is_plot: if True plot the resulting trajectory
        :param fname: demonstrations file
        :param is_plot_robot: if True plot the resulting robot motion
        :param joint_position_limits: joint position limit (scalar)
        :param joint_velocity_limits: joint velocity limit (scalar)

        Return
        ------
        :return: resulting trajectory
        """
        if self.demo_log is None and fname is None:
            raise ValueError('Please generate demo first or give the demo log filename (.npz)')

        nb_dofs = self.gripper.total_links
        if fname is not None:
            demo_log = np.load(fname)
            self.demo_log = dict(demo_log)

        if joint_position_limits is None:
            joint_position_limits = 2 * np.pi - 0.2
        if joint_velocity_limits is None:
            joint_velocity_limits = 10.0

        joint_angle_constraint = JointAngleLimitsConstraint(nb_dofs, joint_position_limits, -joint_position_limits)
        joint_velocity_constraint = JointVelocityLimitsConstraint(nb_dofs, joint_velocity_limits,
                                                                  -joint_velocity_limits)

        ineqn = ListIneqnConstraints([joint_angle_constraint, joint_velocity_constraint])

        joint_space_policy = policy
        joint_space_policy.ineqn = ineqn
        joint_space_policy.eqn = EqnConstraint()

        qt = self.demo_log['q0']
        timesteps = self.demo_log['timesteps']
        xpt_track = self.demo_log['xpt']
        timestamps = np.linspace(0, 1, timesteps)

        qt_track = np.zeros((timesteps, nb_dofs))
        dqt_track = np.zeros((timesteps, nb_dofs))
        xt_track = np.zeros((timesteps, 4))
        dxt_track = np.zeros((timesteps, 4))
        wmat_track = np.zeros((timesteps, sum([skill.dim() for skill in self.skill_list])))
        wmat_full_track = np.zeros((timesteps,
                                    sum([skill.dim() for skill in self.skill_list]),
                                    sum([skill.dim() for skill in self.skill_list])))

        print('testing phase: start generating trajectory ...')

        for i in range(timesteps):
            print('timestamp %1d / %1d' % (i, timesteps), end='\r')
            xt = self.gripper.arm_position_fct(np.expand_dims(qt,axis=0))
            [self.skill_list[si].update_desired_value(self.skill_fcns[si](xt[:2])) for si in range(len(self.skill_list))]

            desired_ = []
            for skill in self.skill_list:
                desired_value = torch.from_numpy(skill.desired_value).double().to(device)
                desired_.append(desired_value)

            desired_ = torch.cat(desired_, dim=-1)
            feat = torch.from_numpy(np.array([timestamps[i]])).double().to(device)
            qt_input = torch.from_numpy(qt).double().to(device)
            qt_input = torch.unsqueeze(qt_input, 0)

            joint_space_policy.ineqn.update(qt_input)
            dq, wmat, ddata = joint_space_policy.forward(feat, qt_input, desired_)
            dq = dq.detach().cpu().numpy()[0, :] * self.demo_log['K'][0, 0] * self.demo_log['dt']
            dqt_track[i, :] = dq
            qt = qt + dq
            qt_track[i, :] = qt
            wmat_track[i, :] = ddata.detach().cpu().numpy()
            wmat_full_track[i, :, :] = wmat.detach().cpu().numpy()
            xt_track[i, :] = self.gripper.compute_ts_arm_fct(qt[np.newaxis, :])

        res = {'xt_track': xt_track,
               'qt_track': qt_track,
               'dxt_track': dxt_track,
               'dqt_track': dqt_track,
               'wmat_track': wmat_track}

        if is_plot_robot:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            color = 'darkblue'
            while True:
                plt.pause(0.01)
                if plt.waitforbuttonpress():
                    break

            ax.plot(pick_goal[0], pick_goal[1], '.', color=[223 / 255, 155 / 255, 27 / 255, 0.8], markersize=60)
            ax.plot(place_goal[0], place_goal[1], '.', color=[162/255, 34/255, 35/255, 0.8], markersize=60)
            ax.plot(xpt_track[:, 0], xpt_track[:, 1], '-.', color=[0, 0, 0, 1], linewidth=2)

            patches_list = []
            for i in range(0, timesteps, 1):
                print('timestamp %1d / %1d' % (i, timesteps), end='\r')
                if patches_list:
                    for patches in patches_list:
                        for patch in patches:
                            patch.remove()
                patches_list = self.gripper.plot(ax, qt_track[i, :], facecolor=color)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.xlim(-20, 20)
                plt.ylim(-20, 20)
                plt.plot(xt_track[:i, 0], xt_track[:i, 1], '-',  color='steelblue', linewidth=2)
                plt.pause(0.01)

            # plt.show()

        if is_plot:
            # xpt = self.demo_log['xpt']
            # plt.figure(figsize=(20, 10))
            #
            # for i in range(2):
            #     plt.plot(np.linspace(0, 1, xpt.shape[0]), xpt[:, i], color='k', linewidth=2, linestyle='-.',
            #              label='desired')
            #     plt.plot(np.linspace(0, 1, xt_track.shape[0]), xt_track[:, i], color=(162/255, 34/255, 35/255), linewidth=2,
            #              label='generated')
            #
            # plt.xlabel('s', fontsize=20)
            # plt.ylabel('position', fontsize=20)
            # plt.xticks(fontsize=20)
            # plt.yticks(fontsize=20)
            # plt.figure(figsize=(20, 10))
            # qt = self.demo_log['qt']
            # if qt is not None:
            #     for i in range(6):
            #         plt.plot(np.linspace(0, 1, qt.shape[0]),qt[:, i], color='k', linewidth=2, linestyle='-.',
            #                  label='desired')
            #         plt.plot(np.linspace(0, 1, qt_track.shape[0]), qt_track[:, i], color=(162/255, 34/255, 35/255), linewidth=2,
            #                  label='generated')
            #
            #     plt.xlabel('s', fontsize=20)
            #     plt.ylabel('joint value', fontsize=20)
            #     plt.xticks(fontsize=20)
            #     plt.yticks(fontsize=20)

            # Weight comparison
            matplotlib.rc_file_defaults()
            wmat_d = self.demo_log['wtraj']
            track_ind = 0
            for i in range(len(self.skill_list)):
                plt.figure(figsize=(10, 5))
                plt.plot(np.linspace(0, 1, wmat_d.shape[0]), wmat_d[:, i], 'k-.', linewidth=2, label='desired')
                plt.plot(np.linspace(0, 1, wmat_track.shape[0]), wmat_track[:, track_ind], color=(162/255, 34/255, 35/255),
                         linestyle='-', linewidth=2, label='generated')
                plt.xlabel(r'$s$', fontsize=32)
                skill_name = self.skill_list[i].name
                plt.ylabel(r'$w_{%s}$' % skill_name, fontsize=32)
                # plt.tight_layout()
                plt.locator_params(axis='x', nbins=3)
                plt.locator_params(axis='y', nbins=3)
                plt.xticks(fontsize=26)
                plt.yticks(fontsize=26)

                track_ind = track_ind + self.skill_list[i].dim()

            plt.show()

        return res
