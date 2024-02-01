import torch
from attrdict import AttrDict

from models.cnp import CNP
from models.modules import BayesianEncoder, DecoderSeparateNetworks

class BAYESCNP(CNP):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid_enc=64,
            dim_hid_dec=128,
            dim_lat=128,
            enc_depth=4,
            dec_depth=3,
            gamma_amortize=False):
        
        super(CNP, self).__init__()

        self.enc1 = BayesianEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid_enc,
                dim_lat=dim_lat,
                depth=enc_depth,
                is_latent=False,
                gamma_amortize=gamma_amortize)

        self.dec = DecoderSeparateNetworks(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_lat*2, # NOTE: pass both {mu_z,cov_z} to decoder
                # dim_enc=dim_lat,
                dim_hid=dim_hid_dec,
                depth=dec_depth)
    
    def predict(self, xc, yc, xt, num_samples=None, max_n_steps=1, min_n_steps=1):
        mu_z, cov_z = self.enc1(xc, yc, max_n_steps, min_n_steps)
        theta = torch.cat([mu_z, cov_z], -1)

        encoded = torch.stack([theta]*xt.shape[-2], -2)

        # encoded_mu = torch.stack([mu_z]*xt.shape[-2], -2)
        # encoded_sigma = torch.stack([cov_z]*xt.shape[-2], -2)

        # return self.dec(encoded_mu, xt, encoded_sigma)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, reduce_ll=True, max_n_steps=1, min_n_steps=1):
        outs = AttrDict()
        py = self.predict(batch.xc, batch.yc, batch.x, max_n_steps=max_n_steps, min_n_steps=min_n_steps)
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
