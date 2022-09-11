import torch
from torch.utils.data import Dataset

from SkillsSequencing.utils.utils import prepare_torch
device = prepare_torch()


class SkillDataset(Dataset):
    """
    This dataset class holds the training data for batch training
    """
    def __init__(self, feat, qt, dqt, desired):
        """
        Initialization of the SkillDataset class.

        Parameters
        ----------
        :param feat: input features (it can be a timestamp or a state.)
        :param qt: joint position
        :param dqt: joint velocity
        :param skills: list of Skill instances
        """
        super(SkillDataset, self).__init__()
        self.feat = torch.from_numpy(feat).to(device)
        self.qt = qt
        self.dqt = dqt
        self.desired_ = desired

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, item):

        return self.feat[item, :], self.qt[item, :],  self.dqt[item, :], self.desired_[item,:]


