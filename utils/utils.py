import argparse
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def parse():
    parser = argparse.ArgumentParser(description="ddn ba")

    parser.add_argument(
        "-s", "--scene", type=str, help="scene to generate", default="fire"
    )

    parser.add_argument(
        "-l", "--length", type=int, help="sequence length", default="1000"
    )

    return parser.parse_args()


def process_poses(poses_in):
    """
    poses_in: 4x4
    poses_out: 0x7
    """
    poses_out = torch.zeros((poses_in.shape[0], 7))
    poses_out[:, 0:3] = poses_in[:, :3, -1]
    for i in range(poses_in.shape[0]):
        p = poses_in[i]
        q = R.from_matrix(p[:3, :3].detach().cpu().numpy()).as_quat()
        poses_out[i, 3:] = torch.from_numpy(q).cuda()
    return poses_out


def cal_trans_rot_error(pred_pose, gt_pose):
    """
    Calculate both translation and rotation errors between two poses.
    :param pred_pose: Predicted pose as [tx, ty, tz, qx, qy, qz, qw]
    :param gt_pose: Ground truth pose as [tx, ty, tz, qx, qy, qz, qw]
    :return: Translation error and rotation error in degrees
    """
    pred_translation = pred_pose[:, :3]
    gt_translation = gt_pose[:, :3]

    pred_R_arr = [
        R.from_quat(pred_pose[i, 3:]).as_matrix() for i in range(len(pred_translation))
    ]
    gt_R_arr = [
        R.from_quat(gt_pose[i, 3:]).as_matrix() for i in range(len(pred_translation))
    ]

    cal_R_arr = [pred_R_arr[i].T @ gt_R_arr[i] for i in range(len(pred_R_arr))]

    r_arr = [R.from_matrix(cal_R_arr[i]).as_rotvec() for i in range(len(cal_R_arr))]
    rotation_error_degs = [
        np.linalg.norm(r_arr[i]) * 180 / np.pi for i in range(len(r_arr))
    ]

    translation_errors = [
        np.linalg.norm(pred_translation[i] - gt_translation[i])
        for i in range(len(pred_translation))
    ]

    return np.mean(translation_errors), np.mean(rotation_error_degs)
