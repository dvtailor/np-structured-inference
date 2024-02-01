import torch
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math


__all__ = ["GPPriorSampler", 'GPSampler', 'RBFKernel', 'PeriodicKernel', 'Matern52Kernel', \
            'WeaklyPeriodicKernel', 'RBFKernelFixedLengthscale', 'Matern52KernelFixedLengthscale', \
            'WeaklyPeriodicKernelFixedLengthscale']


class GPPriorSampler(object):
    """
    Bayesian Optimization에서 이용
    """
    def __init__(self, kernel, t_noise=None):
        self.kernel = kernel
        self.t_noise = t_noise

    # bx: 1 * num_points * 1
    def sample(self, x, device):
        # 1 * num_points * num_points
        cov = self.kernel(x)
        mean = torch.zeros(1, x.shape[1], device=device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        if self.t_noise is not None:
            y += self.t_noise * StudentT(2.1).rsample(y.shape).to(device)

        return y


class GPSampler(object):
    def __init__(self, kernel, t_noise=None, seed=None):
        self.kernel = kernel
        self.t_noise = t_noise
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            device='cpu'):

        batch = AttrDict()
        if isinstance(self.kernel, str): # Mix sine curves
            num_ctx = num_ctx or torch.randint(low=50, high=100, size=[1]).item()
            num_tar = 50
        else:
            num_ctx = num_ctx or torch.randint(low=3, high=max_num_points-3, size=[1]).item()  # Nc
            num_tar = num_tar or torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()  # Nt
        num_points = num_ctx + num_tar  # N = Nc + Nt
        batch.x = x_range[0] + (x_range[1] - x_range[0]) \
                * torch.rand([batch_size, num_points, 1], device=device)  # [B,N,Dx=1]
        batch.xc = batch.x[:,:num_ctx]  # [B,Nc,1]
        batch.xt = batch.x[:,num_ctx:]  # [B,Nt,1]

        # batch_size * num_points * num_points
        # if isinstance(self.kernel, str): # Mix sine curves
        #     # Adapted from: 
        #     #  - https://github.com/google-research/google-research/blob/1ce6116909bb7f868d88e00015ed58837439a08b/ebp/ebp/common/data_utils/curve_reader.py#L135
        #     # max_freq=2
        #     # min_freq=1
        #     # max_scale=1
        #     # min_scale=0.1
        #     # noise_std=0.1
        #     # freq = min_freq + (max_freq-min_freq) \
        #     #                 * torch.rand([batch.x.shape[0], 1, 1], device=device)
        #     # scale = min_scale + (max_scale-min_scale) \
        #     #                 * torch.rand([batch.x.shape[0], 1, 1], device=device)
        #     # cluster = torch.rand([batch_size, num_points, 1], device=device) < 0.5
        #     # y1 = torch.sin(batch.x*freq) * scale + torch.randn_like(batch.x, device=device) * noise_std
        #     # y2 = torch.cos(batch.x*freq) * scale + torch.randn_like(batch.x, device=device) * noise_std

        #     shift = 4*torch.rand([batch.x.shape[0], 1, 1], device=device)-2

        #     freq=2
        #     y1 = torch.sin(batch.x*freq + shift*math.pi)
        #     y2 = torch.cos(batch.x*freq + shift*math.pi)

        #     batch.y = y2 #torch.where(cluster, y1, y2)
        # else:
        cov = self.kernel(batch.x)  # [B,N,N]
        mean = torch.zeros(batch_size, num_points, device=device)  # [B,N]
        batch.y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]

        batch.yc = batch.y[:,:num_ctx]  # [B,Nc,1]
        batch.yt = batch.y[:,num_ctx:]  # [B,Nt,1]

        # if self.t_noise is not None:
        #     if self.t_noise == -1:
        #         t_noise = 0.15 * torch.rand(batch.y.shape).to(device)  # [B,N,1]
        #     else:
        #         t_noise = self.t_noise
        #     batch.y += t_noise * StudentT(2.1).rsample(batch.y.shape).to(device)

        if self.t_noise is not None: # NOTE: noise added only to context
            if self.t_noise == -1:
                t_noise = 0.15 * torch.rand(batch.yc.shape).to(device)  # [B,N,1]
                # t_noise = 0.15 * torch.ones(batch.yc.shape).to(device)  # [B,N,1]
            else:
                t_noise = self.t_noise
            batch.yc += t_noise * StudentT(2.1).rsample(batch.yc.shape).to(device)

        return batch
        # {"x": [B,N,1], "xc": [B,Nc,1], "xt": [B,Nt,1],
        #  "y": [B,N,1], "yc": [B,Nt,1], "yt": [B,Nt,1]}

class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim  [B,N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points  [B,N,N]
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov  # [B,N,N]

# NOTE: fixed length scale implementation need refactoring
class RBFKernelFixedLengthscale(object):
    def __init__(self, noise=1e-5):
        self.length_scale = 1.
        self.noise = noise

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        length = torch.ones([x.shape[0], 1, 1, 1], device=x.device) * self.length_scale
        
        # batch_size * num_points * num_points * dim  [B,N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points  [B,N,N]
        cov = torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.noise * torch.eye(x.shape[-2]).to(x.device)

        return cov  # [B,N,N]


class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov


class Matern52KernelFixedLengthscale(object):
    def __init__(self, noise=1e-5):
        self.length_scale = 0.25
        self.noise = noise

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = self.length_scale * torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        
        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = (1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.noise * torch.eye(x.shape[-2]).to(x.device)

        return cov


class PeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(\
                - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length).pow(2)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class WeaklyPeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        length_periodic = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        length_rbf = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        
        cov = scale.pow(2) * torch.exp(\
                - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length_periodic).pow(2) \
                - 0.5 * torch.pow(dist/length_rbf, 2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov


class WeaklyPeriodicKernelFixedLengthscale(object):
    def __init__(self, noise=1e-5):
        self.length_scale = 1.0
        self.noise = noise

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length_periodic = torch.ones([x.shape[0], 1, 1], device=x.device) * self.length_scale
        length_rbf = torch.ones([x.shape[0], 1, 1, 1], device=x.device) * self.length_scale
        
        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        
        cov = torch.exp(\
                - 2*(torch.sin(0.5*dist.sum(-1))/length_periodic).pow(2) \
                - 0.125 * torch.pow(dist/length_rbf, 2).sum(-1)) \
                + self.noise * torch.eye(x.shape[-2]).to(x.device)

        return cov