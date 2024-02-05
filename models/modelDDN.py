import torch
import numpy as np
import cv2 as cv
import ddn.pytorch.geometry_utilities as geo
from math import degrees

from ddn.pytorch.node import AbstractDeclarativeNode, DeclarativeLayer

from weightedPnP import NonlinearWeightedBlindPnP
from models.matching import Matching
import matplotlib.pyplot as plt

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
            p3d.cpu().numpy(),
            p2d.cpu().numpy(),
            self.K.cpu().numpy(),
            None,
            iterationsCount=1000,
            reprojectionError=1,
        )
        print(retval)
        y = torch.zeros((1, 6)).cuda()
        y[0, :3] = torch.as_tensor(rvet).T
        y[0, 3:] = torch.as_tensor(tvet).T
        return y

    def _run_optimisation(self, *xs, y):
        with torch.enable_grad():
            opt = torch.optim.LBFGS(
                [y],
                lr=1.0,
                max_iter=1000,
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


class DBA(torch.nn.Module):
    def __init__(self):
        super(DBA, self).__init__()
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
        self.ba = BA()
        self.sgMatching = Matching(config)
        self.ransac = self.ba._ransac_p3p
        self.wbpnp = NonlinearWeightedBlindPnP()
        return

    def forward(self, image1, image2, depth1):
        pred = self.sgMatching({"image0": image1, "image1": image2})

        keypoints0 = torch.stack(pred["keypoints0"])
        keypoints1 = torch.stack(pred["keypoints1"])
        matches0 = pred["matches0"]
        # print(matches0)
        matching_scores = pred["matching_scores0"]
        # print(matching_scores.gather(1, torch.argsort(-matching_scores)[:, :100]))
        match_index = torch.argsort(-matching_scores)[:, :30]
        # print(matches0.gather(1, torch.argsort(-matching_scores)[:, :100]))
        # match_index = matching_scores > -0.1
        kp0 = keypoints0.gather(1, match_index.unsqueeze(-1).repeat(1, 1, 2))
        # kp0 = keypoints0[match_index]
        # kp1 = keypoints1[matches0.gather(1, match_index)]
        kp1 = keypoints1.gather(
            1, matches0.gather(1, match_index).unsqueeze(-1).repeat(1, 1, 2)
        )
        depth1 = torch.tensor(depth1, dtype=torch.float32).unsqueeze(0)
        weight = matching_scores.gather(1, match_index)

        kpDepth0 = torch.stack(
            [
                torch.tensor(
                    [getDepthVal(p, depth1[i]) for p in kp0[i]],
                    dtype=torch.float32,
                ).cuda()
                for i in range(depth1.shape[0])
            ]
        ).cuda()

        # kpt1 = [cv.KeyPoint(p[0].item(), p[1].item(), 1) for i, p in enumerate(kp0)]
        # kpt2 = [cv.KeyPoint(p[0].item(), p[1].item(), 1) for i, p in enumerate(kp1)]
        # matchesOpt = [cv.DMatch(i, i, w.item()) for i, w in enumerate(weight)]

        # img1 = cv.imread("./testPictures/frame-001301.color.png")
        # img2 = cv.imread("./testPictures/frame-000014.color.png")
        # outimage = cv.drawMatches(
        #     img1,
        #     kpt1,
        #     img2,
        #     kpt2,
        #     matchesOpt,
        #     outImg=None,
        # )

        # plt.imshow(outimage[:, :, ::-1])
        # plt.show()

        ones = torch.ones((kp0.shape[0], kp0.shape[1], 1)).cuda()

        pt1s = torch.cat((kp0, ones), dim=2)
        p3ds = torch.einsum("jk,b...k->b...j", K_inv, pt1s) * kpDepth0.T

        theta0 = self.ransac(p3ds, kp1)
        theta = self.wbpnp(weight, kp1, p3ds, theta0)
        return theta0, theta


def criterion(y, R_true, t_true):
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
        "rot: {}, trans: {}".format(degrees(error_rotation), error_translation.mean())
    )
    return (
        (error_rotation + 0.25 * error_translation).mean(),
        # (error_rotation + error_translation).mean(),
        error_rotation,
        error_translation,
    )


def test():
    m = DBA().cuda()
    for name, param in m.named_parameters():
        if "superpoint" in name:
            param.requires_grad = False
        # if "superglue" in name:
        #     param.requires_grad = False
    image1 = cv.imread("./testPictures/frame-001301.color.png", 0) / 255
    image2 = cv.imread("./testPictures/frame-000014.color.png", 0) / 255
    depth1 = cv.imread("./testPictures/frame-001301.depth.png", -1) / 1000
    depth2 = cv.imread("./testPictures/frame-000014.depth.png", -1) / 1000

    pose2 = np.loadtxt("./testPoses/frame-001301.pose.txt")
    pose1 = np.loadtxt("./testPoses/frame-000014.pose.txt")

    deltaPose = np.dot(np.linalg.inv(pose1), pose2)
    R_true = torch.tensor(deltaPose[:3, :3], dtype=torch.float32).cuda().unsqueeze(0)
    # rvec = R.from_matrix(deltaPose[:3, :3]).as_rotvec()
    tvec = deltaPose[:3, 3:]
    t_true = torch.tensor(tvec.T, dtype=torch.float32).cuda().unsqueeze(0)

    image1 = (
        torch.as_tensor(image1, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    )
    image2 = (
        torch.as_tensor(image2, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
    )

    opt = torch.optim.Adam(m.parameters(), lr=0.0001)
    opt.zero_grad()

    for i in range(100):
        opt.zero_grad()
        y0, y = m(image1, image2, depth1)
        # error, _, _ = criterion(y0, R_true, t_true)
        error, _, _ = criterion(y, R_true, t_true)
        print(error)
        error.backward()
        opt.step()
    print(y)


if __name__ == "__main__":
    test()
