import pytorch_lightning as pl
import torch
import timm

class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model:int):
        super().__init__()
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            )
        for param in model.parameters():
            param.requires_grad = False

        model.head = torch.nn.Linear(768, d_model)

        self.swinv1 = model

    def forward(self, img, img_mask):
        x = self.swinv1(img)
        img_mask = img_mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]
        return x, img_mask