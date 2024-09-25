import pytorch_lightning as pl
import torch
import timm
from timm.models.swin_transformer import SwinTransformer


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model:int,
                 requires_grad,
                 drop_rate=0.1,
                 proj_drop_rate=0.0,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.,
                 ):
        super().__init__()
        self.swinv1 = SwinTransformer(
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        swin_state_dict = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
        ).state_dict()

        self.swinv1.load_state_dict(swin_state_dict)

        for param in self.swinv1.parameters():
            param.requires_grad = requires_grad

        # add output layer
        self.swinv1.head = torch.nn.Sequential(
            torch.nn.Linear(768, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_rate),
        )

    def forward(self, img, img_mask):
        x = self.swinv1(img)
        img_mask = img_mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]
        return x, img_mask