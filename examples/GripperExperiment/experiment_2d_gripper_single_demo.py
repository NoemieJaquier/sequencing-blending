import os
import inspect
import sys
from optparse import OptionParser
from torch.utils.data import DataLoader

from examples.GripperExperiment.gripper_experiment_classes import Gripper2DExp
from examples.GripperExperiment.gripper_pickplace_skills import *
from SkillsSequencing.skills.skill_classes import *
from SkillsSequencing.qpnet.constraints import TaskSpaceVariableConstraint
from SkillsSequencing.qpnet import qpnet_policies as policy_classes
from SkillsSequencing.qpnet.spec_datasets import SkillDataset
from SkillsSequencing.robots.gripper.gripper2d import Gripper2D

device = prepare_torch()
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = current_dir + '/../../'
os.sys.path.insert(0, '../' + current_dir)


def gripper_experiment(options):
    # Define training parameters
    learning_rate = 0.1
    MAX_EPOCH = 2000
    model_fpath = main_dir + '/' + options.policy_file
    data_file = main_dir + '/' + options.data_file

    if os.path.exists(model_fpath) and options.is_update_policy:
        print('remove old model')
        os.remove(model_fpath)

    # Define current setup
    timesteps = 400
    if options.is_generalized:
        nb_dofs_arm = 10
        nb_dofs_finger = 3
        arm_link_length = 2
        gripper_link_length = 2
        reset_demo_log = True
        q0 = np.array([np.pi/5, np.pi/10, np.pi/10, np.pi/10, np.pi/10, np.pi/20, np.pi/20, np.pi/20, np.pi/20,
                       np.pi/20, np.pi / 2, -np.pi / 3, -np.pi / 6, -np.pi / 2, np.pi / 3, np.pi / 6])
        pick_goal = np.array([-8.5, 8.5])
        place_goal = np.array([-5.5, -8.5])
        pick_ori = [-1.0, 0.0]
        place_ori = [0.0, -1.0]
    else:
        nb_dofs_arm = 4
        nb_dofs_finger = 3
        arm_link_length = 4
        gripper_link_length = 2
        reset_demo_log = False
        q0 = np.array([np.pi / 2, np.pi / 4, 0.2, 0., np.pi / 2, -np.pi / 3, -np.pi / 6, -np.pi / 2, np.pi / 3,
                       np.pi / 6])
        pick_goal, place_goal, pick_ori, place_ori = get_pickplace_parameters()

    # Load arm skills paths
    pick_ds, pick_clf, place_ds, place_clf = get_pickplace_arm_skill_files()

    # Create robot
    robot = Gripper2D(nb_dofs_arm=nb_dofs_arm, nb_dofs_finger=nb_dofs_finger,  arm_link_length=arm_link_length,
                      gripper_link_length=gripper_link_length)

    nb_dofs = nb_dofs_arm + 2 * nb_dofs_finger

    # Create a list of skills
    ts_pos = np.array([[10, 0, 1, 0]])
    pick_pos_skill = TaskspaceSkillBatch(ts_pos, name='pick', compute_fct_batch=robot.compute_ts_arm_fct,
                                         compute_jacobian_fct_batch=robot.compute_ts_jacobian_arm_fct,
                                         config_idx=robot.arm_idx, state_idx=[0, 1, 2, 3])

    ts_pos = np.array([[10, 0, 1, 0]])
    place_pos_skill = TaskspaceSkillBatch(ts_pos, name='place', compute_fct_batch=robot.compute_ts_arm_fct,
                                          compute_jacobian_fct_batch=robot.compute_ts_jacobian_arm_fct,
                                          config_idx=robot.arm_idx, state_idx=[0, 1, 2, 3])

    qclose = np.array([np.pi/2, -np.pi/2, -np.pi/3, -np.pi/2, np.pi/2, np.pi/3])
    close_gripper_skill = JointPositionSkillBatch(qclose, name='close', config_idx=robot.gripper_idx,
                                                  state_idx=[4, 5, 6, 7, 8, 9])

    qopen = np.array([np.pi/2, -np.pi/2, -np.pi/3, -np.pi/2, np.pi/2, np.pi/3])
    open_gripper_skill = JointPositionSkillBatch(qopen, name='open', config_idx=robot.gripper_idx,
                                                 state_idx=[4, 5, 6, 7, 8, 9])

    skill_list = [pick_pos_skill, place_pos_skill, close_gripper_skill, open_gripper_skill]
    skill_fcns = [create_ds_skill('../../' + pick_clf, '../../' + pick_ds, pick_ori, pick_goal, speed=0.15),
                  create_ds_skill('../../' + place_clf, '../../' + place_ds, place_ori, place_goal, speed=0.3),
                  robot.close_skill, robot.open_skill]

    # Define skills clusters
    # One softmax function is defined per skill cluster to guarantee that at least on skill is activated per cluster
    skill_cluster_idx = [0, 2]  # 0, 1 -> arm and 2, 3 -> hand skills

    # Define the loss weights to define the importance of the skills in the loss function
    loss_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Create the experiment
    exp = Gripper2DExp(skill_list, skill_fcns, robot)
    skill_dim = sum([skill_list[i].dim() for i in range(len(skill_list))])
    skill_list, xt_d, dxt_d, desired_, ctrl_idx = exp.get_taskspace_training_data(fname=data_file)
    batch_skill = SkillComplex(skill_dim, nb_dofs, skill_list, skill_cluster_idx=skill_cluster_idx)

    timestamps = np.linspace(0, 1, timesteps - 1)
    feat = timestamps[:, np.newaxis]
    dataset = SkillDataset(feat, xt_d, dxt_d, desired_)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Get the policy type and parameters
    policy_config = {'dim': skill_dim,
                     'fdim': 1,
                     'skill': batch_skill}
    policy_type = getattr(policy_classes, options.policy_name)

    # Define the task space constraints: the task-space variables sharing a control value should be equal
    task_space_variables_constraint = TaskSpaceVariableConstraint(skill_list, ctrl_idx)
    eqn = None
    if task_space_variables_constraint.A is not None:
        eqn = task_space_variables_constraint
    constraints = {'eqn': eqn, 'ineqn': None}

    # Create the policy
    policy = policy_type(**policy_config, **constraints)

    policy.to(device)

    if not os.path.exists(model_fpath):
        # Train the policy
        # If the policy has a full weight matrix, we first train a policy with diagonal weight matrix
        if policy_type == policy_classes.SkillFullyWeightedPolicy:
            diag_policy = policy_classes.SkillDiagonalWeightedPolicy(**policy_config, **constraints)
            diag_policy.to(device)
            diag_policy = policy_classes.train_policy(diag_policy, dataloader, loss_weights=loss_weights,
                                                      model_path=model_fpath,
                                                      learning_rate=learning_rate,
                                                      max_epochs=MAX_EPOCH,
                                                      consider_spec_train=False)
            # The diagonal part of the full weight matrix is then initialized with the pretrained diagonal policy
            # The full weight matrix is then trained starting at this initial point
            policy.QDiagNet = diag_policy.Qnet

        policy = policy_classes.train_policy(policy, dataloader, loss_weights=loss_weights,
                                             model_path=model_fpath,
                                             learning_rate=learning_rate,
                                             max_epochs=MAX_EPOCH,
                                             consider_spec_train=False)

    else:
        # If the policy already exists, load it
        policy.load_policy(model_fpath)
        policy.skill = batch_skill
        policy.to(device)
        if reset_demo_log:
            exp.reset_demo_log_after_ndofs_change(q0)

    # Test the policy
    policy.use_state_space = False
    exp.test_policy(policy, pick_goal=pick_goal, place_goal=place_goal, is_plot=True,
                    is_plot_robot=options.is_show_demo_test)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--policy", dest="policy_name", type="string",
                      default="SkillDiagonalWeightedPolicy",
                      # options='SkillDiagonalWeightedPolicy, SkillFullyWeightedPolicy, '
                      help="Set policy to train or to test. "
                           "Note: the stored policy file should coincide with the policy"
                           "type, otherwise, use --update_policy)")
    parser.add_option("-d", "--data_file", dest="data_file", type="string",
                      default="examples/GripperExperiment/demos/pickplace_demo.npz",
                      help="Set the data filename (to store or update)")
    parser.add_option("-p", "--policy_file", dest="policy_file", type="string",
                      default="examples/GripperExperiment/trained_policies/policy",
                      help="Set the model filename (to store or update)")
    parser.add_option("-u", "--update_policy", action="store_true", dest="is_update_policy", default=True,
                      help="Store (or update) the policy in current folder")
    parser.add_option("--show_demo_test", action="store_true", dest="is_show_demo_test", default=True,
                      help="Show the learned policy")
    parser.add_option("--generalized", action="store_true", dest="is_generalized", default=False,
                      help="Show the learned policy")
    (options, args) = parser.parse_args(sys.argv)

    gripper_experiment(options)
