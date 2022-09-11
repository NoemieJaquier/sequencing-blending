import numpy as np
from SkillsSequencing.skills.mps.dynsys.CLFDS import CLFDS
from SkillsSequencing.skills.mps.dynsys.WSAQF import WSAQF
from SkillsSequencing.skills.mps.dynsys.FNN import SimpleNN
import torch
from SkillsSequencing.utils.utils import prepare_torch
import matplotlib.pyplot as plt

device = prepare_torch()


def get_pickplace_parameters():
    pick_ori = [-1.0, 0.0]
    place_ori = [0.0, -1.0]
    pick_goal = np.array([-8.5, 2.5])
    place_goal = np.array([0.0, -9.5])

    return pick_goal, place_goal, pick_ori, place_ori


def get_pickplace_arm_skill_files():
    pick_clf = 'examples/GripperExperiment/pickplace_ds_skill/pick_posi_clf'
    pick_ds = 'examples/GripperExperiment/pickplace_ds_skill/pick_posi_ds'
    place_clf = 'examples/GripperExperiment/pickplace_ds_skill/place_posi_clf'
    place_ds = 'examples/GripperExperiment/pickplace_ds_skill/place_posi_ds'
    return pick_ds, pick_clf, place_ds, place_clf


def create_ds_skill(clf_file, reg_file, qtarget, goal, speed=0.01):
    clf_model = WSAQF(dim=2, n_qfcn=1)
    reg_model = SimpleNN(2, 2, (20, 20))
    clfds = CLFDS(clf_model, reg_model, rho_0=0.1, kappa_0=0.0001)
    clfds.load_models(clf_file=clf_file, reg_file=reg_file)

    def ds_skill(x):
        x = x[:2]
        d_x = clfds.reg_model.forward(torch.from_numpy(x- goal).to(device))
        d_x = d_x.detach().cpu().numpy()
        x = x + speed * d_x
        x = np.append(x, qtarget)
        return x

    return ds_skill


def close_hand_skill(x):
    return np.array([1, 1])


def open_hand_skill(x):
    return np.array([0, 0])


def check_ds(ax, x0, clf_file, reg_file, qtarget, goal):
    ds = create_ds_skill(clf_file, reg_file, qtarget, goal)
    T = 1000
    test_traj = np.array([x0])
    for _ in range(T):
        x0 = ds(x0)[:2]
        test_traj = np.append(test_traj, np.expand_dims(x0, axis=0), axis=0)
        ax.plot(test_traj[:, 0], test_traj[:, 1], 'b-', linewidth=3)

    plt.show()


if __name__ == "__main__":
    _, ax = plt.subplots(1, 1)
    x0 = np.array([-9.49, 11.2])
    pick_ds, pick_clf, place_ds, place_clf = get_pickplace_arm_skill_files()
    pick_goal, place_goal, pick_ori, place_ori = get_pickplace_parameters()

    check_ds(ax, x0, pick_clf, pick_ds, pick_ori, pick_goal)
