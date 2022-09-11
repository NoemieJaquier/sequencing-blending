# Learning to sequence and blend robotics skills via differentiable optimization
This repository contains code examples for the paper "Learning to sequence and blend robotics skills via differentiable optimization", N. Jaquier, Y. Zhou, J. Sarke, and T. Asfour, Robotics & Automation Letters, 2022.

## Dependencies
This code runs with Python>=3.7. It requires the following packages:
- numpy
- pytorch
- matplotlib
- cvxpy
- qpth
- roboticstoolbox-python

## Installation
To install the library, clone the repository and install the related packages with
```
pip install -r requirements.txt
```

## Examples
### Pick-and-place example with a 4-DoFs 2D gripper, with a single demonstration
The policy with a diagonal weight matrix resulting in Fig. 1b can be reproduced with:
```
python3 experiment_2d_gripper_single_demo.py
```
The policy with a full weight matrix resulting in Fig. 1c can be reproduced with:
```
python3 experiment_2d_gripper_single_demo.py --policy="SkillFullyWeightedPolicy"
```

After training the policy with a diagonal weight matrix, the policy can be applied to a 10-DoFs robot to reproduce Fig. 1e with:
```
python3 experiment_2d_gripper_single_demo.py --update_policy=False --generalized=True
```

You can find trained policies with diagonal (diagonal_policy) and full (full_policy) weight matrices in the folder examples/GripperExperiment/trained_policy.
You can use them by running the experiment files with the following parameters:
```
--update_policy=False --policy_file="examples/GripperExperiment/trained_policies/{diagonal/full}_policy"
```

### Pick-and-place example with a 4-DoFs 2D gripper, with several demonstrations
The policy with a diagonal weight matrix can be trained using 6 demonstrations with:
```
python3 experiment_2d_gripper_several_demos.py
```

After training the policy with a diagonal weight matrix, the results can be reproduced with a 10-DoFs robot with:
```
python3 experiment_2d_gripper_several_demos.py --update_policy=False --generalized=True
```

## Reference
If you found this code useful for you work, we are delighted! Please consider citing the following reference:
```
@article{Jaquier22:RAL,
  author={Jaquier, N. and Zhou, Y. and Starke, J. and Asfour, T.},
  title={Learning to Sequence and Blend Robot Skills via Differentiable Optimization},
  year={2022},
  journal = {{IEEE} Robotics and Automation Letters},
  volume = {7},
  number = {3},
  pages = {8431--8438},
}
```
