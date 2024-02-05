from generates.data_utils import make_dataloader
import torch


def train():
    dataloaders = make_dataloader(None)

    for index1, index2, image1, image2, depth1, depth2, pose1, pose2 in dataloaders[
        "train"
    ]:
        # deltaPose = torch.matmul(torch.linalg.inv(pose1), pose2)
        # R_true = torch.tensor(deltaPose[..., :3, :3])
        # t_true = torch.tensor(deltaPose[..., :3, 3:].transpose(1, 2))


if __name__ == "__main__":
    train()
