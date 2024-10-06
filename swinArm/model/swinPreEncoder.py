import pytorch_lightning as pl
import torch
import timm
from timm.models.swin_transformer import SwinTransformer


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model: int,
                 requires_grad,
                 drop_rate=0.1,
                 proj_drop_rate=0.0,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.,
                 ):
        super().__init__()
        self.swinv1 = SwinTransformer(
            img_size=224,
            in_chans=1,
            embed_dim=48,
            depths=(2, 2, 8, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(28, 24, 7, 7),
            mlp_ratio=2,

            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # add output layer
        self.swinv1.head = torch.nn.Sequential(
            torch.nn.Linear(384, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(drop_rate),
        )
#
    def forward(self, img, img_mask):
        x = self.swinv1(img)
        img_mask = img_mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]
        return x, img_mask
