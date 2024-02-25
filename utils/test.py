from utils import process_poses, cal_trans_rot_error
import torch
import numpy as np
import os


def main():
    fullSeq = os.listdir("/home/david/datasets/pumpkin/test/pose/")
    fullSeq.sort()
    seq1 = fullSeq[1000:]
    pose1 = np.loadtxt("/home/david/datasets/pumpkin/test/pose/000600.pose.txt")
    # pose2 = np.loadtxt("/home/david/datasets/pumpkin/test/pose/frame-003000.pose.txt")
    pose1 = process_poses(torch.tensor(pose1).unsqueeze(0))
    minTrans = 100
    minRot = 100
    filePath = 0
    for p in seq1:
        pose2 = np.loadtxt(os.path.join("/home/david/datasets/pumpkin/test/pose", p))
        pose2 = process_poses(torch.tensor(pose2).unsqueeze(0))
        trans, rot = cal_trans_rot_error(pose1, pose2)
        if rot < minRot:
            filePath = p
            minRot = rot
            minTrans = trans
    print("rot", minTrans, minRot, filePath)
    for p in seq1:
        pose2 = np.loadtxt(os.path.join("/home/david/datasets/pumpkin/test/pose", p))
        pose2 = process_poses(torch.tensor(pose2).unsqueeze(0))
        trans, rot = cal_trans_rot_error(pose1, pose2)
        if trans < minTrans:
            filePath = p
            minRot = rot
            minTrans = trans
    print("trans", minTrans, minRot, filePath)


if __name__ == "__main__":
    main()
