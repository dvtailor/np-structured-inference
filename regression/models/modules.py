import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Gamma, kl_divergence
from torch.distributions.utils import probs_to_logits, logits_to_probs
from models.attention import MultiHeadAttn, SelfAttn
from pyro.distributions import MixtureOfDiagNormals

from utils.sampling import MixtureOfDiagNormalsMod


__all__ = ['PoolingEncoder', 'CrossAttnEncoder', 'Decoder']


def build_mlp(dim_in, dim_hid, dim_out, depth):
    if depth==1:
        modules = [nn.Linear(dim_in, dim_out)] # no hidden layers
    else: # depth>1
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class PoolingEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2):
        super().__init__()

        self.use_lat = dim_lat is not None

        self.net_pre = build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        # MOD: option to not have post-aggregration MLP for deterministic encoder
        if post_depth is not None:
            # assume post_depth >= 2
            self.net_shared = nn.Sequential(*[nn.Linear(dim_hid, dim_hid), nn.ReLU(True)])
            self.net_post = build_mlp(dim_hid, dim_hid,
                    dim_lat if self.use_lat else dim_hid, post_depth-1)
            if self.use_lat:
                self.net_post_sigma = build_mlp(dim_hid, dim_hid, dim_lat, post_depth-1)

    def forward(self, xc, yc, mask=None):
            out = self.net_pre(torch.cat([xc, yc], -1))  # [B,N,Eh]
            if mask is None:
                out = out.mean(-2)  # [B,Eh]
            else:
                mask = mask.to(xc.device)
                out = (out * mask.unsqueeze(-1)).sum(-2) / \
                        (mask.sum(-1, keepdim=True).detach() + 1e-5)
            if self.use_lat:
                out = self.net_shared(out)
                mu = self.net_post(out)
                sigma = self.net_post_sigma(out)
                sigma = 1e-4 + (1.-1e-4) * torch.sigmoid(sigma)
                return Normal(mu, sigma)
            else:
                # MOD
                if hasattr(self, 'net_shared'):
                    out = self.net_shared(out)
                    return self.net_post(out)
                else:
                    return out


class CrossAttnEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            # sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            sigma = 1e-4 + (1.-1e-4) * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out

class NeuCrossAttnEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, w, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)
        v = v * w
        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3, neuboots=False):
        super().__init__()
        self.fc = nn.Linear(dim_x+dim_enc, dim_hid)
        self.dim_hid = dim_hid
        self.neuboots = neuboots

        modules = [nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_y if neuboots else 2*dim_y))
        self.mlp = nn.Sequential(*modules)

    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, x, ctx=None):

        packed = torch.cat([encoded, x], -1)  # [B,(Nbs,)Nt,2Eh+Dx]
        hid = self.fc(packed)  # [B,(Nbs,)Nt,Dh]
        if ctx is not None:
            hid = hid + self.fc_ctx(ctx)  # [B,(Nbs,)Nt,Dh]
        out = self.mlp(hid)  # [B,(Nbs,)Nt,2Dy]
        if self.neuboots:
            return out  # [B,(Nbs,)Nt,2Dy]
        else:
            mu, sigma = out.chunk(2, -1)  # [B,Nt,Dy] each
            sigma = 0.1 + 0.9 * F.softplus(sigma)
            return Normal(mu, sigma)  # Normal([B,Nt,Dy])


class NeuBootsEncoder(nn.Module):

    def   __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2,
            yenc=True, wenc=True, wagg=True):
        super().__init__()

        self.use_lat = dim_lat is not None
        self.yenc = yenc
        self.wenc = wenc
        self.wagg = wagg
        dim_in = dim_x
        if yenc:
            dim_in += dim_y
        if wenc:
            dim_in += 1

        if self.wagg == 'l2a':
            self.agg = nn.Linear(dim_hid,dim_hid)
            self.agg_activation = nn.ReLU()

        self.net_pre = build_mlp(dim_in, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_in, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc=None, w=None):

        device = xc.device
        if not self.yenc:
            _yc = torch.tensor([]).to(device)
        else:
            _yc = yc
        if not self.wenc:
            _w = torch.tensor([]).to(device)
        else:
            _w = w

        # xc: [B,Nbs,N,Dx]
        # yc: [B,Nbs,N,Dy]
        # w: [B,Nbs,N,1]
        """
        Encoder
        """
        input = torch.cat([xc, _yc, _w], -1)  # [B,Nbs,N,?]
        output = self.net_pre(input)  # [B,Nbs,N,Eh]

        """
        Aggregation
        """
        if self.wagg == 'mean':
            out = (output * w).mean(-2)  # [B,Nbs,Eh]
        elif self.wagg == 'max':
            out = (output * w).max(-2).values
        elif self.wagg == 'l2a':
            out = self.agg_activation(self.agg(output * w)).max(dim=-2).values
        else:
            out = output.mean(-2)   # --wagg None
            # [B,Nbs,Eh] : aggregation of context repr

        """
        Decoder
        """
        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)  # [B,Eh]


class CouplingLayer(nn.Module):
  """
  Implementation of the affine coupling layer in RealNVP
  paper.
  """

  def __init__(self, d_inp, d_model, nhead, dim_feedforward, orientation, num_layers):
    super().__init__()

    self.orientation = orientation

    self.embedder = build_mlp(d_inp, d_model, d_model, 2)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    self.ffn = build_mlp(d_model, dim_feedforward, d_inp*2, 2)

    self.scale_net = build_mlp(d_model, dim_feedforward, d_inp, 2)

  def coupling(self, x):
    embeddings = self.embedder(x)
    out_encoder = self.encoder(embeddings)
    s_t = self.ffn(out_encoder)
    scale = torch.sigmoid(self.scale_net(out_encoder))
    return s_t, scale

  def forward(self, x, logdet, invert=False):
    if not invert:
      x1, x2, mask = self.split(x)
      out, scale = self.coupling(x1)
      t, log_s = torch.chunk(out, 2, dim=-1)
      log_s = torch.tanh(log_s) / scale
      s = torch.exp(log_s)
      logdet += torch.sum(log_s.view(s.shape[0], -1), dim=-1)
      y1, y2 = x1, s * (x2 + t)
      return self.merge(y1, y2, mask), logdet

    # Inverse affine coupling layer
    y1, y2, mask = self.split(x)
    out, scale = self.coupling(y1)
    t, log_s = torch.chunk(out, 2, dim=-1)
    log_s = torch.tanh(log_s) / scale
    s = torch.exp(log_s)
    logdet -= torch.sum(log_s.view(s.shape[0], -1), dim=-1)
    x1, x2 = y1, y2 / s - t
    return self.merge(x1, x2, mask), logdet

  def split(self, x):
    assert x.shape[1] % 2 == 0
    mask = torch.zeros(x.shape[1], device='cuda')
    mask[::2] = 1.
    if self.orientation:
      mask = 1. - mask     # flip mask orientation

    x1, x2 = x[:, mask.bool()], x[:, (1-mask).bool()]
    return x1, x2, mask

  def merge(self, x1, x2, mask):
    x = torch.zeros((x2.shape[0], x1.shape[1]*2, x1.shape[2]), device='cuda')
    x[:, mask.bool()] = x1
    x[:, (1-mask).bool()] = x2
    return x

class NICE(nn.Module):
  def __init__(self, d_inp, d_model, nhead, dim_feedforward, num_layers_coupling=2, num_coupling_layers=2):
    super().__init__()

    # alternating mask orientations for consecutive coupling layers
    mask_orientations = [(i % 2 == 0) for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([
        CouplingLayer(
            d_inp, d_model, nhead, dim_feedforward, mask_orientations[i], num_layers_coupling
        ) for i in range(num_coupling_layers)
    ])


  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      return z, log_det_jacobian

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

# nice = NICE(1, 10, 1, 20, 2, 4).cuda()
# y = torch.randn((2, 4, 1), device='cuda')
# z, logdet = nice(y)
# y_prime = nice(z, True)
# print (y)
# print (z)
# print (y_prime)


class BayesianEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, depth=4, is_latent=True, gamma_amortize=False):
        super().__init__()
        self.net_r = build_mlp(dim_x+dim_y, dim_hid, dim_lat, depth)
        self.net_var_r = build_mlp(dim_x+dim_y, dim_hid, dim_lat, depth)
        if gamma_amortize:
            self.net_gam_r = build_mlp(dim_x+dim_y, dim_hid, 1, depth)
        self.dim_lat = dim_lat
        self.is_latent = is_latent
        self.gamma_amortize = gamma_amortize
        self.gm_mom = None
                
    # Implementation adapted from:
    # https://github.com/michaelvolpp/neural_process/blob/ec4f68110e4d73aff800231e90300add8a89e3a3/src/neural_process/aggregator.py#L177
    def forward(self, xc, yc, max_n_steps=1, min_n_steps=1, fixed_point_threshold=1e-4, eval_elbo=False):
        def _cavi_gaussian(priors, potentials, q_gamma):
            mu_z, _, _ = priors # NB: mu_z, gam_r, gam_prior
            r, var_r = potentials
            a_new,b_new,c_new,d_new = q_gamma

            gm_mom = a_new/b_new # gamma 1st moment
            gm_mom_prior = c_new/d_new
            # Bayesian context aggregation modified
            v = r - mu_z[None, None, :]
            cov_w_inv = 1 / var_r
            cov_z_new = 1 / (gm_mom_prior[:,None] + torch.sum(gm_mom[:,:,None] * cov_w_inv, dim=1))
            mu_z_new = mu_z + cov_z_new * torch.sum(gm_mom[:,:,None] * cov_w_inv * v, dim=1)

            return mu_z_new, cov_z_new

        def _cavi_gamma(priors, potentials, q_gauss):
            _, gam_r, gam_prior = priors
            r, var_r = potentials
            mu_z_new, cov_z_new = q_gauss

            a_new = gam_r + self.dim_lat/2
            gm_rate_update = (torch.sum(r**2/var_r, dim=-1)
                            - 2*torch.sum((r*mu_z_new[:,None,:])/var_r, dim=-1)
                            + torch.sum(mu_z_new[:,None,:]**2/var_r, dim=-1)
                            + torch.sum(cov_z_new[:,None,:]/var_r, dim=-1))
            b_new = gam_r + gm_rate_update/2

            c_new = gam_prior + self.dim_lat/2
            d_new = gam_prior + torch.sum(mu_z_new**2 + cov_z_new, dim=-1)/2

            return a_new,b_new,c_new,d_new

        def _elbo(priors, potentials, q_gauss, q_gamma):
            mu_z, gam_r, gam_prior = priors
            r, var_r = potentials
            mu_z_new, cov_z_new = q_gauss
            a_new,b_new,c_new,d_new = q_gamma

            gm_rate_update = (torch.sum(r**2/var_r, dim=-1)
                            - 2*torch.sum((r*mu_z_new[:,None,:])/var_r, dim=-1)
                            + torch.sum(mu_z_new[:,None,:]**2/var_r, dim=-1)
                            + torch.sum(cov_z_new[:,None,:]/var_r, dim=-1))

            # elbo of surrogate VI objective; scaled by n_data
            # NB: potential bug, no dependence on mu_z (likely assumed zero mean like in paper)
            elbo_new = (torch.sum(-self.dim_lat * torch.log(torch.tensor(2.*math.pi))
                    + self.dim_lat * (torch.digamma(a_new) - torch.log(b_new))
                    - torch.sum(torch.log(var_r), dim=2)
                    - (a_new/b_new) * gm_rate_update, dim=1)/2
                    - kl_divergence(Gamma(a_new,b_new), Gamma(gam_r, gam_r)).sum(1)
                    - kl_divergence(Gamma(c_new,d_new),Gamma(gam_prior,gam_prior))
                    + (-self.dim_lat * torch.log(torch.tensor(2.*math.pi)) 
                    + self.dim_lat * (torch.digamma(torch.tensor(c_new)) - torch.log(d_new)) 
                    - (c_new/d_new) * torch.sum(mu_z_new**2 + cov_z_new, dim=-1))/2
                    + Normal(mu_z_new, cov_z_new.sqrt()).entropy().sum(-1))/xc.shape[1]

            return elbo_new
        
        # cavi termination from Tipping paper (based on moments)
        def _check_termination(q_gauss_old, q_gauss_new, q_gamma_old, q_gamma_new):
            mu_z_old, _ = q_gauss_old
            mu_z_new, _ = q_gauss_new
            a_old,b_old,c_old,d_old = q_gamma_old
            a_new,b_new,c_new,d_new = q_gamma_new

            diff_gauss = torch.abs(mu_z_old-mu_z_new).mean().item()
            diff_gam_1 = torch.abs(torch.log(a_old/b_old) - torch.log(a_new/b_new)).mean().item()
            diff_gam_2 = torch.abs(torch.log(c_old/d_old) - torch.log(c_new/d_new)).mean().item()

            return (diff_gauss < fixed_point_threshold) and \
                    (diff_gam_1 < fixed_point_threshold) and \
                    (diff_gam_2 < fixed_point_threshold)

        r = self.net_r(torch.cat([xc, yc], -1))
        var_r = self.net_var_r(torch.cat([xc, yc], -1))
        var_r = 1e-4 + (1.-1e-4) * torch.nn.Sigmoid()(var_r) # Rectification proposed in [Lee et. al., 2018] may have undesirable behaviour in BCA 
        
        is_elbo_track = (max_n_steps > min_n_steps)
    
        if self.gamma_amortize:
            gam_r = self.net_gam_r(torch.cat([xc, yc], -1)).squeeze()
            gam_r = 1. + 1e-4 + (1.-1e-4) * torch.nn.Sigmoid()(gam_r) # must be >1 for valid T-dist
        else:
            # fixed prior inspired by [Tipping & Lawrence, 2005] adjusted by latent_dim
            gam_r = torch.ones(xc.shape[:2]).to(xc.device) * 0.01 * self.dim_lat

        # prior
        mu_z = torch.zeros((self.dim_lat,)).to(xc.device)
        gam_prior = 1e-6 * self.dim_lat # same setting as [Tipping & Lawrence, 2005] adjusted by latent_dim
        # gam_prior = 1e-4 * self.dim_lat # less broad prior

        # initial setting of gamma posterior (obs noise)
        a_new = torch.ones(xc.shape[:2]).to(xc.device)
        b_new = torch.ones(xc.shape[:2]).to(xc.device)

        # initial setting of gamma posterior (hierarchical prior)
        # i.e. recovers in n_steps=1, var_z = 1
        c_new = torch.ones((xc.shape[0],)).to(xc.device)
        d_new = torch.ones((xc.shape[0],)).to(xc.device)

        priors = (mu_z, gam_r, gam_prior)
        potentials = (r, var_r)
        q_gamma = (a_new,b_new,c_new,d_new)
        q_gauss = _cavi_gaussian(priors, potentials, q_gamma)
        q_gamma = _cavi_gamma(priors, potentials, q_gauss)

        if max_n_steps > 1:

            if not is_elbo_track: # const num iters
                for step in range(2,max_n_steps+1):
                    q_gauss = _cavi_gaussian(priors, potentials, q_gamma)
                    q_gamma = _cavi_gamma(priors, potentials, q_gauss)
                q_gauss_new = q_gauss
                q_gamma_new = q_gamma

            else: # fixed point: early termination
                #  i.e. backprop through only single step
                # elbo_old = torch.ones(xc.shape[:1]).to(xc.device)*torch.inf
                # for step in range(2,max_n_steps+1):
                #     if step >= min_n_steps:
                #         elbo_new = _elbo(priors, potentials, q_gauss, q_gamma)
                #         elbo_diff = torch.abs(elbo_new - elbo_old).mean()
                #         if elbo_diff.item() < fixed_point_threshold:
                #             break
                #         elbo_old = elbo_new
                #     q_gamma = _cavi_gamma(priors, potentials, q_gauss)
                #     q_gauss = _cavi_gaussian(priors, potentials, q_gamma)
                # q_gauss_new = q_gauss

                # with torch.no_grad():
                q_gauss_old = q_gauss
                q_gamma_old = q_gamma
                for step in range(2,max_n_steps+1):
                    q_gauss_new = _cavi_gaussian(priors, potentials, q_gamma_old)
                    q_gamma_new = _cavi_gamma(priors, potentials, q_gauss_new)
                    if (step >= min_n_steps) and _check_termination(q_gauss_old, q_gauss_new, q_gamma_old, q_gamma_new):
                        break
                    q_gauss_old = q_gauss_new
                    q_gamma_old = q_gamma_new

                # use implicit diff too
                # q_gauss_new = _cavi_gaussian(priors, potentials, (q_gamma_new[0],q_gamma_new[1].detach(),q_gamma_new[2],q_gamma_new[3].detach()))
                # q_gauss_new = _cavi_gaussian(priors, potentials, (q_gamma_new[0],q_gamma_new[1],q_gamma_new[2],q_gamma_new[3]))

                # # approx. to implicit diff via "1st order Neumann series expansion" (backprop single step)
                # q_gamma = _cavi_gamma(priors, potentials, (q_gauss[0].detach(), q_gauss[1].detach()))
                # q_gauss_new = _cavi_gaussian(priors, potentials, q_gamma)

        else:
            q_gauss_new = q_gauss
            q_gamma_new = q_gamma

        mu_z_new, cov_z_new = q_gauss_new
        if self.is_latent:
            if eval_elbo:
                elbo = _elbo(priors, potentials, q_gauss_new, q_gamma_new)
                return Normal(mu_z_new, cov_z_new.sqrt()), elbo    
            else:
                return Normal(mu_z_new, cov_z_new.sqrt())
        else:
            return mu_z_new, cov_z_new

# # Adapted from:
# # https://github.com/michaelvolpp/neural_process/blob/ec4f68110e4d73aff800231e90300add8a89e3a3/src/np_util/output_trafo.py
# def output_trafo(output, lower_bound=1e-4):
#     softplus_stiffness = 1.0
#     # return lower_bound + torch.nn.Softplus(beta=softplus_stiffness)(output) # original from github link
#     # Use form proposed in [Le et. al., 2018]
#     return lower_bound + (1.-lower_bound) * torch.nn.Softplus(beta=softplus_stiffness)(output)

# # https://github.com/michaelvolpp/neural_process/blob/ec4f68110e4d73aff800231e90300add8a89e3a3/src/neural_process/decoder_network.py#L161
# def parametrize_latent_cov(cov):
#     safe_log = 1e-8
#     cov = cov + safe_log
#     parametrized_cov = torch.log(cov)
#     return parametrized_cov


class DecoderSeparateNetworks(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3):
        super().__init__()        
        self.mlp_mu = build_mlp(dim_x+dim_enc, dim_hid, dim_y, depth)
        self.mlp_sigma = build_mlp(dim_x+dim_enc, dim_hid, dim_y, depth)

    def forward(self, encoded_mu, x, encoded_sigma=None): # NOTE: can remove encoded_sigma if no longer difference between mu/sigma decoder
        packed = torch.cat([encoded_mu, x], -1)
        mu = self.mlp_mu(packed)

        if encoded_sigma is not None:
            packed = torch.cat([encoded_sigma, x], -1)
        
        sigma = self.mlp_sigma(packed)
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        
        return Normal(mu, sigma)


class BayesianEncoderMoG(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, depth=4, n_components=3, is_latent=True):
        super().__init__()
        self.net_r = build_mlp(dim_x+dim_y, dim_hid, dim_lat, depth)
        self.net_var_r = build_mlp(dim_x+dim_y, dim_hid, dim_lat, depth)
        self.dim_lat = dim_lat
        self.n_components = n_components
        ## Prior parameters for MoG ##
        self.mu_z = nn.Parameter( 0.1 * torch.randn(n_components, dim_lat), requires_grad=False)
        # Pre-sigmoid prior covariance (prior variances effectively initialised to 1)
        self.cov_z_ = nn.Parameter( torch.logit(torch.ones(n_components,dim_lat), eps=1e-6), requires_grad=False )
        # Pre-softmax mixing proportions
        self.pi_ = nn.Parameter( probs_to_logits(torch.ones(self.n_components) / self.n_components), requires_grad=False )
        self.is_latent = is_latent

    def forward(self, xc, yc, num_samples=None):
        r = self.net_r(torch.cat([xc, yc], -1))
        var_r = self.net_var_r(torch.cat([xc, yc], -1))
        var_r = 1e-4 + (1.-1e-4) * nn.Sigmoid()(var_r)

        cov_z = nn.Sigmoid()(self.cov_z_) # prior cov

        # update for gauss moments (same as BA)
        v = r.unsqueeze(1) - self.mu_z[None,:,None,:]
        cov_w_inv = 1 / var_r.unsqueeze(1)
        prec_z_new = (cov_z**-1).unsqueeze(0) + torch.sum(cov_w_inv, dim=-2)
        cov_z_new = 1 / prec_z_new
        mu_z_new = self.mu_z[None,:] + cov_z_new * torch.sum(cov_w_inv * v, dim=-2)

        log_Ck = -((torch.log(cov_z).unsqueeze(0) - torch.log(cov_z_new)).sum(dim=-1) 
                    + torch.sum(self.mu_z**2/cov_z, dim=-1).unsqueeze(0)
                    - torch.sum(mu_z_new**2 * prec_z_new, dim=-1))/2.

        logits_new = self.pi_.unsqueeze(0).expand(xc.shape[0],-1) + log_Ck
        logits_new = logits_new - torch.logsumexp(logits_new, 1).unsqueeze(1)

        locs = mu_z_new
        coord_scale = cov_z_new.sqrt()
        component_logits = logits_new

        # add num_samples to dist batch_shape to allow multiple samples
        if num_samples is not None:
            locs = locs.unsqueeze(0).expand(num_samples,*locs.shape)
            coord_scale = coord_scale.unsqueeze(0).expand(num_samples,*coord_scale.shape)
            component_logits = component_logits.unsqueeze(0).expand(num_samples,*component_logits.shape)

        if self.is_latent:
            return MixtureOfDiagNormalsMod(locs, coord_scale, component_logits)
            # return MixtureOfDiagNormals(locs, coord_scale, component_logits)
        else:
            return mu_z_new, cov_z_new, logits_to_probs(logits_new)
