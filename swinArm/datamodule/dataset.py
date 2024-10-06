import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from .transforms import (ScaleAugmentation,
                         ScaleToLimitRange,
                         rand_aug)

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        trans_list.append(tr.ToTensor())

        # if is_train and scale_aug:
        #     trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        if is_train and scale_aug:
            trans_list.append(
                tr.RandomChoice(
                    [rand_aug,
                     nn.Identity(),
                     ], p=[0.2, 0.8])
            )

        trans_list += [
            # tr.Normalize(mean=[0.485, 0.456, 0.406],
            #              std=[0.229, 0.224, 0.225]),
            tr.Resize((224, 224)),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        fname, img, caption = self.ds[idx]

        img = [self.transform(im) for im in img]

        return fname, img, caption

    def __len__(self):
        return len(self.ds)
