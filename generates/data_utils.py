from torch.utils import data
import cv2 as cv
from itertools import permutations
from torch.utils.data import (
    DataLoader,
    Dataset,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
import os
import torch
import pickle
import numpy as np


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


class SevenScenes(Dataset):
    def __init__(self, picklePath) -> None:
        super().__init__()
        self.picklePath = picklePath
        with open(picklePath, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return self.data[ndx]


def make_collate_fn(dataset):
    def collate_fn(data_list):
        index = [item["id"] for item in data_list]
        index = list(permutations(index, 2))
        pair = list(set(map(tuple, map(sorted, index))))

        image1 = []
        image2 = []

        depth1 = []
        depth2 = []

        pose1 = []
        pose2 = []

        index1 = []
        index2 = []

        for p in pair:
            if dataset[p[1]]["id"] in dataset[p[0]]["positives"]:
                image1.append(
                    torch.tensor(
                        # cv.imread(dataset[p[0]]["rgb"], 0) / 255,
                        cv.imread(dataset[p[0]]["rgb"]),
                        dtype=torch.float32,
                        device="cuda",
                    ).unsqueeze(0)
                )
                image2.append(
                    torch.tensor(
                        # cv.imread(dataset[p[1]]["rgb"], 0) / 255,
                        cv.imread(dataset[p[1]]["rgb"]),
                        dtype=torch.float32,
                        device="cuda",
                    ).unsqueeze(0)
                )
                depth1.append(
                    torch.tensor(
                        cv.imread(dataset[p[0]]["depth"], -1) / 1000,
                        dtype=torch.float32,
                        device="cuda",
                    ).unsqueeze(0)
                )
                depth2.append(
                    torch.tensor(
                        cv.imread(dataset[p[1]]["depth"], -1) / 1000,
                        dtype=torch.float32,
                        device="cuda",
                    ).unsqueeze(0)
                )
                pose1.append(
                    torch.tensor(
                        np.loadtxt(dataset[p[0]]["pose"]),
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
                pose2.append(
                    torch.tensor(
                        np.loadtxt(dataset[p[1]]["pose"]),
                        dtype=torch.float32,
                        device="cuda",
                    )
                )

                index1.append(torch.tensor(p[0]).cuda())
                index2.append(torch.tensor(p[1]).cuda())

        if len(index1) == 0:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return (
            torch.stack(index1),
            torch.stack(index2),
            torch.stack(image1),
            torch.stack(image2),
            torch.stack(depth1),
            torch.stack(depth2),
            torch.stack(pose1),
            torch.stack(pose2),
        )

    return collate_fn


def make_dataloader(params):
    datasets = {}
    dataloaders = {}
    samplers = {}
    collate_fns = {}
    datasets["train"] = SevenScenes(
        "/home/david/Code/ddn_pose_estimation/fire_train.pickle"
    )
    datasets["test"] = SevenScenes(
        "/home/david/Code/ddn_pose_estimation/fire_test.pickle"
    )

    samplers["train"] = BatchSampler(
        sampler=RandomSampler(range(len(datasets["train"]))),
        batch_size=12,
        drop_last=False,
    )

    samplers["test"] = BatchSampler(
        sampler=RandomSampler(range(len(datasets["test"]))),
        batch_size=12,
        drop_last=False,
    )

    collate_fns["train"] = make_collate_fn(datasets["train"])
    collate_fns["test"] = make_collate_fn(datasets["test"])

    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_sampler=samplers["train"],
        collate_fn=collate_fns["train"],
    )

    dataloaders["test"] = DataLoader(
        datasets["test"], batch_sampler=samplers["test"], collate_fn=collate_fns["test"]
    )

    return dataloaders


def test():
    dataloaders = make_dataloader(None)
    # testDataset = SevenScenes("/home/david/Code/ddn_pose_estimation/fire_train.pickle")
    # print(testDataset[10])

    for index1, index2, image1, image2, depth1, depth2, pose1, pose2 in dataloaders[
        "train"
    ]:
        print(image1.shape)


if __name__ == "__main__":
    test()
