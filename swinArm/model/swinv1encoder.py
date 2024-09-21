import pytorch_lightning as pl
import torch
from .swinmodule.swin_transformer import SwinTransformer

class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 img_size:int,
                 in_chans:int,
                 embed_dim:int,
                 depth,
                 num_heads,
                 window_size:int,
                 d_model:int,
                 drop_rate:float):
        super().__init__()
        self.swinv1 = SwinTransformer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            d_model=d_model,
            drop_rate=drop_rate
        )
    def forward(self, img, img_mask):
        return self.swinv1(img, img_mask)