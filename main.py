import torch
import numpy as np

import cv2.cv2 as cv

from matplotlib import pyplot as plt
from ddn.pytorch.node import AbstractDeclarativeNode


class Epipolar(AbstractDeclarativeNode):
    def __init__(self):
        super().__init__()
        self.K = torch.Tensor(
            [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]
        )

    def solve(self, p1, p2):
        p1 = p1.detach()
        p2 = p2.detach()


def pixel2cam(p, k):
    return np.array([(p[0] - k[0, 2]) / k[0, 0], (p[1] - k[1, 2]) / k[1, 1]])


def getDepthVal(depth_img, p):
    return depth_img[int(p[1]), int(p[0])]


def toIntList(l):
    l = [int(i) for i in l]
    return l


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert batch of 3D angle-axis vectors into a batch of 3D rotation matrices

    Arguments:
        angle_axis: (b, 3) Torch tensor,
            batch of 3D angle-axis vectors

    Return Values:
        rotation_matrix: (b, 3, 3) Torch tensor,
            batch of 3D rotation matrices
    """
    assert (
        angle_axis.shape[-1] == 3
    ), "Angle-axis vector must be a (*, 3) tensor, received {}".format(angle_axis.shape)

    def angle_axis_to_rotation_matrix_rodrigues(angle_axis, theta2):
        theta = torch.sqrt(theta2).unsqueeze(-1)  # bx1
        r = angle_axis / theta  # bx3
        rx = r[..., 0]  # b
        ry = r[..., 1]  # b
        rz = r[..., 2]  # b
        r_skew = torch.zeros_like(r).unsqueeze(-1).repeat_interleave(3, dim=-1)  # bx3x3
        r_skew[..., 2, 1] = rx
        r_skew[..., 1, 2] = -rx
        r_skew[..., 0, 2] = ry
        r_skew[..., 2, 0] = -ry
        r_skew[..., 1, 0] = rz
        r_skew[..., 0, 1] = -rz
        R = (
            torch.eye(3, dtype=r.dtype, device=r.device).unsqueeze(0)
            + theta.sin().unsqueeze(-1) * r_skew
            + (1.0 - theta.cos().unsqueeze(-1)) * torch.matmul(r_skew, r_skew)
        )  # bx3x3
        return R

    def angle_axis_to_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=-1)
        ones = torch.ones_like(rx)
        R = torch.cat([ones, -rz, ry, rz, ones, -rx, -ry, rx, ones], dim=1).view(
            -1, 3, 3
        )
        return R

    theta2 = torch.einsum("bi,bi->b", (angle_axis, angle_axis))

    eps = 1e-6
    if (theta2 > eps).all():
        rotation_matrix = angle_axis_to_rotation_matrix_rodrigues(angle_axis, theta2)
    else:
        rotation_matrix = angle_axis_to_rotation_matrix_taylor(angle_axis)
        rotation_matrix_rodrigues = angle_axis_to_rotation_matrix_rodrigues(
            angle_axis, theta2
        )
        # Iterate over batch dimension
        # Note: cannot use masking or torch.where because any NaNs in the gradient
        # of the unused branch get propagated
        # See: https://github.com/pytorch/pytorch/issues/9688
        for b in range(angle_axis.shape[0]):
            if theta2[b, ...] > eps:
                rotation_matrix[b, ...] = rotation_matrix_rodrigues[b : (b + 1), ...]
    return rotation_matrix


def project_3d_2d(p3d, y, K):
    r = angle_axis_to_rotation_matrix(y[:, :3])

    P2 = torch.einsum("bjk, b...k->b...j", r, p3d) + y[:, 3:]
    X2 = torch.einsum("jk, b...k->b...j", K, P2)
    X2 = X2 / X2[:, -1].unsqueeze(-1)
    return X2


def J(p3d, p2d, y, k):
    projected_points = project_3d_2d(
        torch.as_tensor(p3d, dtype=torch.float32),
        y,
        torch.as_tensor(k, dtype=torch.float32),
    )[:, :2]
    error = torch.sum(
        (projected_points - torch.as_tensor(p2d, dtype=torch.float32)) ** 2, dim=-1
    )
    return torch.sum(error)


def run_optimize(p3d, p2d, y, k):
    with torch.enable_grad():
        opt = torch.optim.LBFGS(
            [y],
            lr=0.001,
            max_iter=1000,
            max_eval=None,
            tolerance_grad=1e-40,
            tolerance_change=1e-40,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def reevaluate():
            opt.zero_grad()
            f = J(p3d, p2d, y, k).sum()  # sum over batch elements
            f.backward()
            return f

        opt.step(reevaluate)
    return y


def main():
    img1 = cv.imread("./1.png")
    img2 = cv.imread("./2.png")

    depth_img1 = cv.imread("./1_depth.png", 0) / 5000
    depth_img2 = cv.imread("./2_depth.png", 0) / 5000

    K = np.array([[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])
    K = torch.as_tensor(K, dtype=torch.float32)
    k_inv = np.linalg.inv(K)
    print(k_inv)
    orb = cv.ORB().create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    min_dist, max_dist = 100000, 0

    for m in matches:
        dist = m.distance
        min_dist = dist if dist < min_dist else min_dist
        max_dist = dist if dist > max_dist else max_dist
    matches_opt = []
    print("min: ", min_dist)
    print("max: ", max_dist)
    for m in matches:
        if m.distance < max(2 * min_dist, 30.0):
            matches_opt.append(m)
    print("num: ", len(matches_opt))
    print(matches_opt[0])
    print(type(matches_opt[0]))
    # outimage = cv.drawMatches(img1, kp1, img2, kp2, matches_opt, outImg=None)
    # plt.imshow(outimage[:, :, ::-1])
    # plt.show()
    points1, points2 = [], []
    for m in matches_opt:
        points1.append(list(kp1[m.queryIdx].pt))
        points2.append(list(kp2[m.trainIdx].pt))

    # em, mask = cv.findEssentialMat(np.array(points1), np.array(
    #     points2), K, cv.RANSAC, 0.999, 1.0, np.array([]))
    # print("EssentialMatrix: ", em)
    # num, R, t, _ = cv.recoverPose(em, np.array(points1), np.array(points2), K)
    # print("R, t: ", R, t)
    # t_x = np.array([[0.0, -t[2][0], t[1][0]],
    #                [t[2][0], 0, -t[0][0]], [-t[1][0], t[0][0], 0]])
    # print("t_x: ", t_x)

    # for i in range(len(points1)):
    #     pt1 = pixel2cam(points1[i], K)
    #     pt2 = pixel2cam(points2[i], K)
    #     pt1 = np.array([pt1[0], pt1[1], 1])
    #     pt2 = np.array([pt2[0], pt2[1], 1])
    #     print(np.dot(np.dot(np.dot(pt2.T, t_x), R), pt1))

    depth = []
    valid_points = []
    for i in range(len(points1)):
        tmp = getDepthVal(depth_img1, points1[i])
        if tmp == 0.0:
            valid_points.append(False)
        else:
            depth.append(tmp)
            valid_points.append(True)
    points1 = np.array(points1)[valid_points]
    points2 = np.array(points2)[valid_points]
    ones = np.ones((len(points1), 1))
    pt1s = np.concatenate((np.array(points1), ones), axis=1)

    p3d = np.einsum("jk,...k->...j", k_inv, pt1s) * np.array([depth]).T
    print(p3d)
    retval, rvet, tvet, inliers = cv.solvePnPRansac(
        p3d, np.array(points2), K.numpy(), None
    )
    y = torch.zeros((1, 6))
    y[0, :3] = torch.as_tensor(rvet).T
    y[0, 3:] = torch.as_tensor(tvet).T * 0.01
    y = y.requires_grad_(True)
    rmat, jaco = cv.Rodrigues(rvet)
    print("rotation mat from Rodrigues \n", rmat)
    r = angle_axis_to_rotation_matrix(torch.as_tensor(rvet).view((1, -1)))
    print("rotation mat from torch \n", r)

    # P2 = np.einsum('jk, ...k->...j', rmat, p3d) + tvet.T
    # X2 = np.einsum('jk, ...k->...j', K, P2)
    # X2 /= X2.T[-1].reshape((-1, 1))
    p3d = torch.as_tensor(p3d, dtype=torch.float32)
    # X2 = project_3d_2d(p3d, y, K)
    print(y)
    y = run_optimize(p3d, points2, y, K)
    print(y)
    outimage = img1
    # print(X2)
    # print(X2.shape)
    # print(np.mean(np.sum((points2[:, :2] - X2[:, :2])**2, -1)))
    for p in points2:
        outimage = cv.circle(outimage, toIntList(tuple(p)), 2, (255.0, 0.0, 0.0), 3)

    # for p in X2:
    #     outimage = cv.circle(outimage, toIntList(
    #         tuple(p[:2])), 2, (0.0, 255.0, 0.0), 3)

    # plt.imshow(outimage)
    # plt.show()

    return


if __name__ == "__main__":
    main()
