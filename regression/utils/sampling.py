import torch
from pyro.distributions import MixtureOfDiagNormals

def gather(items, idxs):
    K = idxs.shape[0]
    idxs = idxs.to(items[0].device)
    gathered = []
    for item in items:
        gathered.append(torch.gather(
            torch.stack([item]*K), -2,
            torch.stack([idxs]*item.shape[-1], -1)).squeeze(0))
    return gathered[0] if len(gathered) == 1 else gathered

def sample_subset(*items, r_N=None, num_samples=None):
    r_N = r_N or torch.rand(1).item()
    K = num_samples or 1
    N = items[0].shape[-2]
    Ns = min(max(1, int(r_N * N)), N-1)
    batch_shape = items[0].shape[:-2]
    idxs = torch.rand((K,)+batch_shape+(N,)).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(*items, num_samples=None, r_N=1.0, N_s=None):
    K = num_samples or 1
    N = items[0].shape[-2]
    N_s = N_s or max(1, int(r_N * N))
    batch_shape = items[0].shape[:-2]
    idxs = torch.randint(N, size=(K,)+batch_shape+(N_s,))
    return gather(items, idxs)

def sample_mask(B, N, num_samples=None, min_num=3, prob=0.5):
    min_num = min(min_num, N)
    K = num_samples or 1
    fixed = torch.ones(K, B, min_num)
    if N - min_num > 0:
        rand = torch.bernoulli(prob*torch.ones(K, B, N-min_num))
        mask = torch.cat([fixed, rand], -1)
        return mask.squeeze(0)
    else:
        return fixed.squeeze(0)


# Copied from https://docs.pyro.ai/en/1.5.1/_modules/pyro/distributions/diag_normal_mixture.html#MixtureOfDiagNormals
# No backprop through categorical (numerical issues)
class MixtureOfDiagNormalsMod(MixtureOfDiagNormals):
    def rsample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            which = self.categorical.sample(sample_shape)
        locs = self.locs
        scales = self.coord_scale
        # component_logits = self.component_logits
        # pis = self.categorical.probs
        noise_shape = sample_shape + self.locs.shape[:-2] + (self.dim,)

        dim = scales.size(-1)
        white = locs.new(noise_shape).normal_()
        n_unsqueezes = locs.dim() - which.dim()
        for _ in range(n_unsqueezes):
            which = which.unsqueeze(-1)
        which_expand = which.expand(tuple(which.shape[:-1] + (dim,)))
        loc = torch.gather(locs, -2, which_expand).squeeze(-2)
        sigma = torch.gather(scales, -2, which_expand).squeeze(-2)
        z = loc + sigma * white

        return z