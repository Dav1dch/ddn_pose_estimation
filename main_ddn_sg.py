import torch
import numpy as np
import sys
import cv2 as cv
import ddn.pytorch.geometry_utilities as geo
from ddn.pytorch.node import AbstractDeclarativeNode, DeclarativeLayer
from scipy.spatial.transform import Rotation as R


from math import degrees

from models.matching import Matching


K = torch.tensor([[532.0, 0, 320.0], [0.0, 531.0, 240.0], [0.0, 0.0, 1.0]]).cuda()
K_inv = torch.linalg.inv(K)


def getDepthVal(p, depthImage):
    return depthImage[int(p[1]), int(p[0])]


class BA(AbstractDeclarativeNode):
    def __init__(self):
        super().__init__()
        self.Kv = torch.tensor([[585.0, 585.0, 320.0, 240.0]]).cuda()
        self.K = torch.tensor(
            [[532.0, 0, 320.0], [0.0, 531.0, 240.0], [0.0, 0.0, 1.0]]
        ).cuda()

    def objective(self, p3d, p2d, w, y):
        projectedPoint = geo.project_points_by_theta(p3d, y, self.Kv)
        # print(projectedPoint[0][0])
        # print(p2d[0][0])
        squared_error = torch.sum((projectedPoint - p2d) ** 2, dim=-1)
        w = torch.nn.functional.relu(w)  # Enforce non-negative weights
        return torch.einsum("bn,bn->b", (w, squared_error))

    def solve(self, p3d, p2d, w):
        p3d = p3d.detach()
        p2d = p2d.detach()
        w = w.detach()
        y = self._ransac_p3p(p3d, p2d).requires_grad_()
        y = self._run_optimisation(p3d, p2d, w, y=y)
        return y.detach(), None

    def _ransac_p3p(self, p3d, p2d):
        retval, rvet, tvet, inliers = cv.solvePnPRansac(
            p3d.cpu().numpy(), p2d.cpu().numpy(), self.K.cpu().numpy(), None
        )
        y = torch.zeros((1, 6)).cuda()
        y[0, :3] = torch.as_tensor(rvet).T
        y[0, 3:] = torch.as_tensor(tvet).T
        return y

    def _run_optimisation(self, *xs, y):
        with torch.enable_grad():
            opt = torch.optim.LBFGS(
                [y],
                lr=0.1,
                max_iter=100,
                max_eval=None,
                tolerance_grad=1e-40,
                tolerance_change=1e-40,
                history_size=100,
                line_search_fn="strong_wolfe",
            )

        def reevaluate():
            opt.zero_grad()
            f = self.objective(*xs, y=y).sum()  # sum over batch elements
            f.backward()
            return f

        opt.step(reevaluate)
        return y


y_ = None


def main():
    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": -1,
        },
        "superglue": {
            "weights": "indoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        },
    }

    sgMatching = Matching(config).cuda()

    image1 = cv.imread("./frame-001301.color.png", 0) / 255
    image2 = cv.imread("./frame-000014.color.png", 0) / 255
    depth1 = cv.imread("./frame-001301.depth.png", -1) / 1000
    depth2 = cv.imread("./frame-000014.depth.png", -1) / 1000

    pose2 = np.loadtxt("./frame-001301.pose.txt")
    pose1 = np.loadtxt("./frame-000014.pose.txt")

    deltaPose = np.dot(np.linalg.inv(pose1), pose2)
    R_true = torch.tensor(deltaPose[:3, :3]).cuda()
    rvec = R.from_matrix(deltaPose[:3, :3]).as_rotvec()
    tvec = deltaPose[:3, 3:]
    t_true = torch.tensor(tvec.T).cuda()

    image1 = (
        torch.as_tensor(image1, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    )
    image2 = (
        torch.as_tensor(image2, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    )

    pred = sgMatching({"image0": image1, "image1": image2})

    keypoints0 = pred["keypoints0"][0]
    keypoints1 = pred["keypoints1"][0]
    matches0 = pred["matches0"]
    matching_scores = pred["matching_scores0"]
    match_index = matching_scores > 0.7
    kp0 = keypoints0[match_index.squeeze()]
    kp1 = keypoints1[matches0[match_index]]
    weight = matching_scores[match_index]
    kpDepth0 = torch.tensor(
        [getDepthVal(p, depth1) for p in kp0], dtype=torch.float32
    ).cuda()
    valid = kpDepth0 != 0

    kp0 = kp0[valid]
    kp1 = kp1[valid]
    kpt1 = [cv.KeyPoint(p[0].item(), p[1].item(), 1) for i, p in enumerate(kp0)]
    kpt2 = [cv.KeyPoint(p[0].item(), p[1].item(), 1) for i, p in enumerate(kp1)]
    weight = weight[valid]
    kpDepth0 = kpDepth0[valid]
    matchesOpt = [cv.DMatch(i, i, w.item()) for i, w in enumerate(weight)]
    import matplotlib.pyplot as plt

    #

    node = BA()

    ones = torch.ones((kp0.shape[0], 1)).cuda()

    pt1s = torch.cat((kp0, ones), dim=1)
    p3ds = torch.einsum("jk,...k->...j", K_inv, pt1s) * kpDepth0.unsqueeze(0).T
    weight_orig = weight.clone()
    y_ddn, _ = torch.no_grad()(node.solve)(
        p3ds.unsqueeze(0), kp1.unsqueeze(0), weight.unsqueeze(0)
    )

    y_origin = y_ddn.clone()

    # Define the upper-level objective:
    def J(p3d, q2d, weight, y):
        """Compute sum of angular and positional camera errors"""
        # if y is None:
        #     y, _ = torch.no_grad()(node.solve)(p, q, w)
        R_ = geo.angle_axis_to_rotation_matrix(y[..., :3])
        t = y[..., 3:]
        max_dot_product = 1.0 - 1e-7
        error_rotation = (
            (0.5 * ((R_ * R_true).sum(dim=(-2, -1)) - 1.0))
            .clamp_(-max_dot_product, max_dot_product)
            .acos()
        )
        error_translation = (t - t_true).norm(dim=-1)
        print(
            "rot: {:0.2f}, trans: {:0.6f}".format(
                degrees(error_rotation[0, ...]), error_translation[0, ...]
            )
        )
        return (
            (error_rotation + 0.25 * error_translation).mean(),
            error_rotation,
            error_translation,
        )

    weight = weight_orig.detach().unsqueeze(0).requires_grad_()
    y = y_origin.clone()

    ba_declarative_layver = DeclarativeLayer(node)
    optimizer = torch.optim.LBFGS(
        [weight], lr=1, max_iter=50, line_search_fn="strong_wolfe"
    )
    global y_

    def reevaluate():
        global y_
        optimizer.zero_grad()
        y = ba_declarative_layver(p3ds.unsqueeze(0), kp1.unsqueeze(0), weight)
        y_ = y.detach()
        z, error_rotation, error_translation = J(
            p3ds.unsqueeze(0), kp1.unsqueeze(0), weight, y
        )
        z.backward()
        # history_loss.append(z.clone())
        # history_rot.append(degrees(error_rotation[0, ...]))  # First batch element only
        # history_tran.append(error_translation[0, ...])  # First batch element only
        return z

    # optimizer.step(reevaluate)

    retval, rvet, tvet, inliers = cv.solvePnPRansac(
        p3ds.cpu().numpy(), kp1.cpu().numpy(), K.cpu().numpy(), None
    )
    y = torch.zeros((1, 6)).cuda()
    y[0, :3] = torch.as_tensor(rvet).T
    y[0, 3:] = torch.as_tensor(tvet).T

    R_ = geo.angle_axis_to_rotation_matrix(y[..., :3])
    t = y[..., 3:]
    max_dot_product = 1.0 - 1e-7
    error_rotation = (
        (0.5 * ((R_ * R_true).sum(dim=(-2, -1)) - 1.0))
        .clamp_(-max_dot_product, max_dot_product)
        .acos()
    )
    error_translation = (t - t_true).norm(dim=-1)
    print(
        "rot: {:0.2f}, trans: {:0.6f}".format(
            degrees(error_rotation[0, ...]), error_translation[0, ...]
        )
    )
    # y = y.requires_grad_(True)

    projectedPoint = geo.project_points_by_theta(
        p3ds.unsqueeze(0), y_, torch.tensor([[532.0, 531.0, 320.0, 240.0]]).cuda()
    )

    kpt2 = [
        cv.KeyPoint(p[0].item(), p[1].item(), 1)
        for i, p in enumerate(projectedPoint[0])
    ]
    outImage = cv.drawMatches(
        (image1.squeeze().squeeze().cpu().numpy() * 255).astype(np.uint8),
        kpt1,
        (image2.squeeze().squeeze().cpu().numpy() * 255).astype(np.uint8),
        kpt2,
        matchesOpt,
        outImg=None,
    )
    plt.imshow(outImage[:, :, ::-1])
    plt.show()

    # print(projectedPoint)
    # print(projectedPoint.shape)
    outImage = (image2.squeeze().squeeze().cpu().numpy() * 255).astype(np.uint8)
    outImage = cv.merge((outImage, outImage, outImage))
    for p in kp1:
        outImage = cv.circle(
            outImage, list(map(int, p.cpu().numpy())), 3, (255, 255, 0)
        )
    for p in projectedPoint.squeeze():
        outImage = cv.circle(
            outImage, list(map(int, p.cpu().numpy())), 3, (0, 255, 255)
        )
    for i in range(len(projectedPoint[0])):
        outImage = cv.line(
            outImage,
            list(map(int, kp1[i].cpu().numpy())),
            list(map(int, projectedPoint[0][i].cpu().numpy())),
            (0, 0, 255),
        )
    plt.imshow(outImage[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    main()
