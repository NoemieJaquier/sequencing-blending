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
    num_data_file = 8
    model_fpath = main_dir + '/' + options.policy_file
    data_file = main_dir + '/' + options.data_file
    num_demos = options.num_demos
    num_tests = options.num_tests
    # Train and test sets
    train_ids = [0, 1, 2, 3, 4, 5]
    test_ids = [6, 7]

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
    else:
        nb_dofs_arm = 4
        nb_dofs_finger = 3
        arm_link_length = 4
        gripper_link_length = 2
        reset_demo_log = False
        q0 = np.array([np.pi / 2, np.pi / 4, 0.2, 0., np.pi / 2, -np.pi / 3, -np.pi / 6, -np.pi / 2, np.pi / 3,
                       np.pi / 6])

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

    # Define skills clusters
    # One softmax function is defined per skill cluster to guarantee that at least on skill is activated per cluster
    skill_cluster_idx = [0, 2]  # 0, 1 -> arm and 2, 3 -> hand skills

    # Define the loss weights to define the importance of the skills in the loss function
    loss_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Setup the training trajectories
    datasets = []
    batch_skills = []
    exps = []

    # Create skills, data, experiments for all available data file (incl. train and test files)
    for i in range(num_data_file):
        data_file_demo = data_file + str(i) + '.npz'
        demo_log = dict(np.load(data_file_demo))

        # Skills for the given pick and place positions
        pick_goal = demo_log['pick_goal']
        place_goal = demo_log['place_goal']
        pick_pos = pick_goal[:2]
        pick_ori = pick_goal[2:]
        place_pos = place_goal[:2]
        place_ori = place_goal[2:]
        skill_fcns = [create_ds_skill('../../' + pick_clf, '../../' + pick_ds, pick_ori, pick_pos, speed=0.15),
                      create_ds_skill('../../' + place_clf, '../../' + place_ds, place_ori, place_pos, speed=0.3),
                      robot.close_skill, robot.open_skill]

        # Create the experiment
        exp = Gripper2DExp(skill_list, skill_fcns, robot)
        skill_dim = sum([skill_list[i].dim() for i in range(len(skill_list))])
        skill_list, xt_d, dxt_d, desired_, ctrl_idx = exp.get_taskspace_training_data(fname=data_file_demo)
        batch_skill = SkillComplex(skill_dim, nb_dofs, skill_list, skill_cluster_idx=skill_cluster_idx)

        timestamps = np.linspace(0, 1, timesteps - 1)
        feat = timestamps[:, np.newaxis]
        dataset = SkillDataset(feat, xt_d, dxt_d, desired_)

        exps.append(exp)
        datasets.append(dataset)
        batch_skills.append(batch_skill)

    # Get the policy type and parameters
    policy_config = {'dim': skill_dim,
                     'fdim': 1,
                     'skill': batch_skill}
    policy_type = getattr(policy_classes, options.policy_name)
    num_epochs_per_demo_pass = int(MAX_EPOCH/num_demos/10)  # 10 passes through each demo

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
            # Pre-train the policy using all demonstrations
            epochs = 0
            while epochs < MAX_EPOCH:
                # We do num_epochs_per_demo_pass for each demo, looping this until the maximum of epochs is reached
                for i in range(num_demos):
                    id = train_ids[i]
                    dataset = datasets[id]
                    batch_skill = batch_skills[id]
                    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
                    diag_policy.skill = batch_skill

                    diag_policy = policy_classes.train_policy(diag_policy, dataloader, loss_weights=loss_weights,
                                                              model_path=model_fpath,
                                                              learning_rate=learning_rate,
                                                              max_epochs=num_epochs_per_demo_pass,
                                                              consider_spec_train=False)
                epochs += num_demos * num_epochs_per_demo_pass
            # The diagonal part of the full weight matrix is then initialized with the pretrained diagonal policy
            # The full weight matrix is then trained starting at this initial point
            policy.QDiagNet = diag_policy.Qnet

        # Train the policy using all demonstrations
        epochs = 0
        while epochs < MAX_EPOCH:
            # We do num_epochs_per_demo_pass for each demo, looping this until the maximum of epochs is reached
            for i in range(num_demos):
                id = train_ids[i]
                dataset = datasets[id]
                batch_skill = batch_skills[id]
                dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
                policy.skill = batch_skill
                policy = policy_classes.train_policy(policy, dataloader, loss_weights=loss_weights,
                                                     model_path=model_fpath,
                                                     learning_rate=learning_rate,
                                                     max_epochs=num_epochs_per_demo_pass,
                                                     consider_spec_train=False)
            epochs += num_demos * num_epochs_per_demo_pass

    else:
        # If the policy already exists, load it
        policy.load_policy(model_fpath)
        policy.skill = batch_skill
        policy.to(device)

    # Test the policy
    policy.use_state_space = False
    for i in range(num_tests):
        id = test_ids[i]
        exp = exps[id]
        # Pick and place locations
        data_file_demo = data_file + str(id) + '.npz'
        demo_log = dict(np.load(data_file_demo))
        pick_goal = demo_log['pick_goal']
        place_goal = demo_log['place_goal']
        if reset_demo_log:
            exp.reset_demo_log_after_ndofs_change(q0)
        exp.test_policy(policy, pick_goal=pick_goal, place_goal=place_goal, is_plot=True,
                        is_plot_robot=options.is_show_demo_test)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--policy", dest="policy_name", type="string",
                      default="SkillDiagonalWeightedPolicy",
                      # options='SkillDiagonalWeightedPolicy, '
                      help="Set policy to train or to test. "
                           "Note: the stored policy file should coincide with the policy"
                           "type, otherwise, use --update_policy)")
    parser.add_option("-d", "--data_file", dest="data_file", type="string",
                      default="examples/GripperExperiment/demos/pickplace_demo",
                      help="Set the data filename (to store or update)")
    parser.add_option("-p", "--policy_file", dest="policy_file", type="string",
                      default="examples/GripperExperiment/trained_policies/diagonal_policy_several_demos",
                      help="Set the model filename (to store or update)")
    parser.add_option("-u", "--update_policy", action="store_true", dest="is_update_policy", default=False,
                      help="Store (or update) the policy in current folder")
    parser.add_option("--show_demo_test", action="store_true", dest="is_show_demo_test", default=True,
                      help="Show the learned policy")
    parser.add_option("--generalized", action="store_true", dest="is_generalized", default=False,
                      help="Show the learned policy")
    parser.add_option("-n", "--num_demos", dest="num_demos", type="int", default=6)
    parser.add_option("-t", "--num_tests", dest="num_tests", type="int", default=1)
    (options, args) = parser.parse_args(sys.argv)

    gripper_experiment(options)
