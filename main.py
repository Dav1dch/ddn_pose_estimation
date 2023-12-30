import torch
import numpy as np

import cv2.cv2 as cv

from matplotlib import pyplot as plt
from ddn.pytorch.node import AbstractDeclarativeNode


class Epipolar(AbstractDeclarativeNode):
    def __init__(self):
        super().__init__()
        self.K = torch.Tensor(
            [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])

    def solve(self, p1, p2):
        p1 = p1.detach()
        p2 = p2.detach()


def pixel2cam(p, k):
    return np.array([(p[0] - k[0, 2]) / k[0, 0], (p[1] - k[1, 2]) / k[1, 1]])


def getDepthVal(depth_img, p):
    return depth_img[int(p[1]), int(p[0])]


def main():
    img1 = cv.imread('./1.png')
    img2 = cv.imread('./2.png')

    depth_img1 = cv.imread('./1_depth.png', 0) / 5000
    depth_img2 = cv.imread('./2_depth.png', 0) / 5000

    K = np.array(
        [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]])
    k_inv = np.linalg.inv(K)
    orb = cv.ORB().create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(des1, des2)
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
    # outimage = cv.drawMatches(img1, kp1, img2, kp2, matches_opt, outImg=None)
    # plt.imshow(outimage[:, :, ::-1])
    # plt.show()
    points1, points2 = [], []
    for m in matches_opt:
        points1.append(list(kp1[m.queryIdx].pt))
        points2.append(list(kp2[m.trainIdx].pt))
    em, mask = cv.findEssentialMat(np.array(points1), np.array(
        points2), K, cv.RANSAC, 0.999, 1.0, np.array([]))
    print("EssentialMatrix: ", em)
    num, R, t, _ = cv.recoverPose(em, np.array(points1), np.array(points2), K)
    print("R, t: ", R, t)
    t_x = np.array([[0.0, -t[2][0], t[1][0]],
                   [t[2][0], 0, -t[0][0]], [-t[1][0], t[0][0], 0]])
    print("t_x: ", t_x)

    # for i in range(len(points1)):
    #     pt1 = pixel2cam(points1[i], K)
    #     pt2 = pixel2cam(points2[i], K)
    #     pt1 = np.array([pt1[0], pt1[1], 1])
    #     pt2 = np.array([pt2[0], pt2[1], 1])
    #     print(np.dot(np.dot(np.dot(pt2.T, t_x), R), pt1))

    depth = []
    for i in range(len(points1)):
        depth.append(getDepthVal(depth_img1, points1[i]))
    print(depth)
    ones = np.ones((len(points1), 1))
    pt1s = np.concatenate((np.array(points1), ones), axis=1)

    p3d = np.einsum('jk,...k->...j', k_inv, pt1s) * np.array([depth]).T
    print(p3d * np.array([depth]).T)
    retval, rvet, tvet, inliers = cv.solvePnPRansac(
        p3d, np.array(points2), K, None)
    rmat, jaco = cv.Rodrigues(rvet)

    P2 = np.einsum('jk, ...k->...j', rmat, p3d)
    X2 = np.einsum('jk, ...k->...j', K, P2)
    print(X2 / X2.T[-1].reshape((-1, 1)))
    print(points2)

    return


if __name__ == '__main__':
    main()
