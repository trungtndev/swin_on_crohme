import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from swinArm.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
import tqdm
import gc
from .vocab import vocab

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory

# load data
def data_iterator(
        data: Data,
        batch_size: int,
        batch_Imagesize: int = MAX_SIZE,
        maxlen: int = 200,
        maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """

    with open(f"{archive}/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in tqdm.tqdm(captions):
        tmp = line.strip().split()
        img_name = tmp[0]
        formula = tmp[1:]

        img_name = img_name.split(".")[0]
        with open(f"{archive}/{dir_name}/img/{img_name}.bmp", "rb") as f:
            img = Image.open(f).convert("RGB").copy()
            img = img.resize((224, 224))
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}/{archive}, with data size: {len(data)}")
    gc.collect()
    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 3, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
            dataset_name: str = "",
            test_year: str = "2014",
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            scale_aug: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.dataset_name = dataset_name
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, f"{self.dataset_name}/train", self.train_batch_size),
                True,
                self.scale_aug,
            )
            self.val_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, f"{self.dataset_name}/{self.test_year}", self.eval_batch_size),
                False,
                self.scale_aug,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, f"{self.dataset_name}/{self.test_year}", self.eval_batch_size),
                False,
                self.scale_aug,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )