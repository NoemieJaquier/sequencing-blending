from typing import Union, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.path as mpath
import matplotlib.patches as mpatches


def plot_planar_robot(ax,
                      joint_angles: np.ndarray, link_lengths: Union[int, np.ndarray],
                      width_param: float = 0.5, facecolor: Union[str, list] = 'darkblue',
                      edgecolor: Union[str, list] = 'white') \
        -> list:
    """
    This function displays a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure
    :param joint_angles: joint angles (numpy array of size nb_dofs)
    :param link_lengths: link lengths (scalar if all links have the same length, numpy array of size nb_dofs otherwise)

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot [base_patches, link0_patches, ..., linkn_patches]

    Notes
    -------
    To erase the robot from the figure do:
    for patch_list in patch_list_planar_robot:
    for patch in patch_list:
        patch.remove()

    """
    # Make background white
    ax.patch.set_facecolor('white')
    # No grid
    ax.grid(False)
    # Patch list to return
    patch_list = []

    # Number of DoFs
    nb_dofs = joint_angles.shape[0]
    # Links length as an array
    if np.isscalar(link_lengths):
        link_lengths = np.tile(link_lengths, nb_dofs)

    # Plot basis
    patch_list_basis = plot_robot_basis(ax, width_param, facecolor, edgecolor)
    patch_list.append(patch_list_basis)

    # Plot links
    current_endeff_position = np.zeros(2)
    for i in range(len(joint_angles)):
        patch_list_link, current_endeff_position = plot_robot_link(ax, np.sum(joint_angles[:i+1]), link_lengths[i],
                                                                   current_endeff_position, width_param, facecolor,
                                                                   edgecolor)
        patch_list.append(patch_list_link)

    return patch_list


def plot_robot_basis(ax, width_param: float = 0.05,
                     facecolor: Union[str, list] = 'gray', edgecolor: Union[str, list] = 'white') \
        -> list:
    """
    This function displays the basis of a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot base [base_patch, line0_patch, ..., line5_patch]
    """

    nb_segments = 30
    width_param = width_param * 1.2

    # Draw basis
    # Define contour
    t1 = np.linspace(0, np.pi, nb_segments - 2)
    x = np.zeros((nb_segments, 2))
    x[:, 0] = np.append(np.append(width_param * 1.5, width_param * 1.5 * np.cos(t1)), -width_param * 1.5)
    x[:, 1] = np.append(np.append(-width_param * 1.2, width_param * 1.5 * np.sin(t1)), - width_param * 1.2)
    # Draw path
    path = mpath.Path(x)
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=1)
    ax.add_patch(patch)

    # Patch list to return
    patch_list = [patch]

    # Draw 5 bottom lines
    # Lines coordinates
    x2 = np.zeros((5, 2))
    x2[:, 0] = np.linspace(-width_param * 1.2, width_param * 1.2, 5)
    x2[:, 1] = -width_param * 1.2
    x3 = x2 + np.tile(0.25*np.array([-0.5, -1]), (5, 1))
    # Define and draw paths
    for i in range(5):
        x_path = np.array([[x2[i, 0], x2[i, 1]], [x3[i, 0], x3[i, 1]]])
        path = mpath.Path(x_path)
        patch_line = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=facecolor, linewidth=2)
        ax.add_patch(patch_line)
        patch_list.append(patch_line)

    return patch_list


def plot_robot_link(ax,
                    angle: float, link_length: int, base_position: np.ndarray, width_param: float = 0.05,
                    facecolor: Union[str, list] = 'gray', edgecolor: Union[str, list] = 'white') \
        -> Tuple[list, np.ndarray]:
    """
    This function displays a link of a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure
    :param angle: angle of the link to display
    :param link_length: length of the link
    :param base_position: position of the base of the link (numpy array of size 2)

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot link [link_patch, circle0_patch, circle1_patch]
    :return endeffector_position: position of the end of the link (numpy array of size 2)
    """
    # Number of segments
    nb_segments = 30

    # Draw link "bar"
    t1 = np.linspace(0, -np.pi, int(nb_segments/2))
    t2 = np.linspace(np.pi, 0, int(nb_segments/2))
    # Define contour
    x = np.zeros((nb_segments, 2))
    x[:, 0] = np.append(width_param * np.sin(t1), link_length + width_param * np.sin(t2))
    x[:, 1] = np.append(width_param * np.cos(t1), width_param * np.cos(t2))
    x = np.vstack((x, x[0, :]))  # Add first element at the end to have a closed contour
    # Rotate contours
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x = np.dot(R, x.T).T + base_position
    # Draw path
    path = mpath.Path(x)
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(patch)

    # Draw holes
    # Define circle contour
    endeff_position = np.dot(R, np.array([link_length, 0])) + base_position
    msh = np.zeros((nb_segments, 2))
    msh[:, 0] = np.sin(np.linspace(0, 2 * np.pi, nb_segments))
    msh[:, 1] = np.cos(np.linspace(0, 2 * np.pi, nb_segments))
    msh *= width_param * 0.2
    # Draw first circle
    path = mpath.Path(msh + base_position)
    patch_circle1 = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(patch_circle1)
    # Draw second circle
    path = mpath.Path(msh + endeff_position)
    patch_circle2 = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(patch_circle2)

    # Patch list to return
    patch_list = [patch, patch_circle1, patch_circle2]

    return patch_list, endeff_position
