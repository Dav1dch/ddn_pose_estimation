from generates.data_utils import make_dataloader
from utils.utils import process_poses, cal_trans_rot_error
from utils.loss import PoseLoss
from models.modelVIT import ViT, ViTSP
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def iter(model, dataloader, Test, criterion, opt):
    if Test:
        model.eval()
    else:
        model.train()
    trans_error_total = []
    rot_error_total = []
    loss_total = []
    for index1, index2, image1, image2, depth1, depth2, pose1, pose2 in tqdm(
        dataloader
    ):
        if index1 == None:
            continue
        opt.zero_grad()
        deltaPose = torch.matmul(torch.linalg.inv(pose1), pose2)
        deltaPose = process_poses(deltaPose).cuda()

        # R_true = torch.tensor(deltaPose[..., :3, :3])
        # t_true = torch.tensor(deltaPose[..., :3, 3:].transpose(1, 2))
        img1 = rearrange(image1, "b n h w c -> b (c n) h w")
        img2 = rearrange(image2, "b n h w c -> b (c n) h w")
        pred = model(img1, img2)

        pos_pred = pred[:, :3]
        ori_pred = pred[:, 3:]
        pos_true = deltaPose[:, :3]
        ori_true = deltaPose[:, 3:]
        ori_pred = F.normalize(ori_pred, p=2, dim=1)
        ori_true = F.normalize(ori_true, p=2, dim=1)
        # pos_pred, ori_pred, pos_true, ori_true
        loss_pose, loss_pos, loss_ori = criterion(
            pos_pred, ori_pred, pos_true, ori_true
        )
        pred_pose = np.hstack(
            (pos_pred.detach().cpu().numpy(), ori_pred.detach().cpu().numpy())
        )
        true_pose = np.hstack(
            (
                pos_true.detach().cpu().numpy(),
                ori_true.detach().cpu().numpy(),
            )
        )

        trans_error, rot_error = cal_trans_rot_error(
            # pred_pose, gt_pose.detach().cpu().numpy()
            pred_pose,
            true_pose,
        )
        trans_error_total.append(trans_error)
        rot_error_total.append(rot_error)
        loss_total.append(loss_pose.item())
        if not Test:
            loss_pose.backward()
            opt.step()
    if Test:
        print(
            "Test",
            np.mean(trans_error_total),
            " ",
            np.mean(rot_error_total),
            " ",
            np.mean(loss_total),
        )
    else:
        print(
            "Train",
            np.mean(trans_error_total),
            " ",
            np.mean(rot_error_total),
            " ",
            np.mean(loss_total),
        )


def train():
    dataloaders = make_dataloader(None)

    model = ViTSP(
        image_size=(480, 640),
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    ).cuda()

    criterion = PoseLoss(learn_beta=False).cuda()
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 0.0005}])
    for i in range(300):
        iter(model, dataloaders["train"], False, criterion, opt)
        iter(model, dataloaders["test"], True, criterion, opt)


if __name__ == "__main__":
    train()
