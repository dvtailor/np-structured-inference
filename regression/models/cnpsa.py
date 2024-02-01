import torch
import torch.nn as nn
from attrdict import AttrDict

from models.modules import PoolingEncoder, Decoder, DecoderSeparateNetworks

class CNPSA(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.enc1 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                self_attn=True,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        # self.enc2 = PoolingEncoder(
        #         dim_x=dim_x,
        #         dim_y=dim_y,
        #         dim_hid=dim_hid,
        #         pre_depth=enc_pre_depth,
        #         post_depth=enc_post_depth)

        # self.dec = Decoder(
        #         dim_x=dim_x,
        #         dim_y=dim_y,
        #         # dim_enc=2*dim_hid,
        #         dim_enc=dim_hid,
        #         dim_hid=dim_hid,
        #         depth=dec_depth)

        self.dec = DecoderSeparateNetworks(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, num_samples=None):
        # encoded = torch.cat([self.enc1(xc, yc), self.enc2(xc, yc)], -1)
        encoded = self.enc1(xc, yc)
        encoded = torch.stack([encoded]*xt.shape[-2], -2)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        py = self.predict(batch.xc, batch.yc, batch.x)
        ll = py.log_prob(batch.y).sum(-1)

        if self.training:
            outs.loss = -ll.mean()
        else:
            se = torch.sum((batch.y - py.loc)**2, dim=-1)
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
                outs.ctx_se = se[...,:num_ctx].mean().sqrt()
                outs.tar_se = se[...,num_ctx:].mean().sqrt()
            else:
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]

        return outs
