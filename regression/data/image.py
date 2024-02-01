import torch
import torchvision
from attrdict import AttrDict
from torch.distributions import StudentT
import numpy as np
import skimage as sk
from torchvision.transforms import InterpolationMode

def img_to_task(img, num_ctx=None,
        max_num_points=None, target_all=False, t_noise=None, plot_noise_type=None):

    B, C, H, W = img.shape
    num_pixels = H*W
    img = img.view(B, C, -1)

    # if t_noise is not None:
    #     if t_noise == -1:
    #         t_noise = 0.09 * torch.rand(img.shape)
    #     img += t_noise * StudentT(2.1).rsample(img.shape)

    batch = AttrDict()
    max_num_points = max_num_points or num_pixels
    num_ctx = num_ctx or \
            torch.randint(low=3, high=max_num_points-3, size=[1]).item()
    num_tar = max_num_points - num_ctx if target_all else \
            torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()
    num_points = num_ctx + num_tar
    idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(img.device)
    batch.y = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(img.device)

    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]
    batch.yc = batch.y[:,:num_ctx]
    batch.yt = batch.y[:,num_ctx:]

    # Only add t-noise to context set
    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.09 * torch.rand(batch.yc.shape).to(img.device)
            batch.yc += t_noise * StudentT(2.1).rsample(batch.yc.shape).to(img.device)
        else: # assuming t_noise = {1,2,3,4,5}
            noise_funcs = [gaussian_noise,shot_noise,impulse_noise,speckle_noise]
            if plot_noise_type is None:
                idx = torch.randint(high=4,size=(1,)).item()
            else:
                idx = plot_noise_type
            f = noise_funcs[idx]
            batch_yc_npy = batch.yc.detach().cpu().numpy()
            batch_yc_npy_noise = f(batch_yc_npy, t_noise)
            batch.yc = torch.from_numpy(batch_yc_npy_noise).float().to(img.device)

    return batch

# def get_subscaled_img_data(img, W_new):
#     B, C, H, W = img.shape
#     num_pixels = W_new*W_new
#     # NOTE: may want to check interpolation method for subscaling
#     img_subscale = torchvision.transforms.functional.resize(img, (W_new,W_new))

#     idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1).to(img.device)
#     x1, x2 = idxs//W_new, idxs%W_new
#     factor = W//W_new
#     x1 *= factor # transform to original grid
#     x2 *= factor
#     xc = torch.stack([
#         2*x1.float()/(H-1) - 1,
#         2*x2.float()/(W-1) - 1], -1).to(img.device)

#     img_subscale = img_subscale.view(B, C, -1)
#     yc = (torch.gather(img_subscale, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
#             .transpose(-2, -1) - 0.5).to(img.device)
#     return xc, yc

# # super-resolution version
# #  - change default behaviour to predict all pixels 
# def img_to_task_sr(img, max_num_points=None, target_all=True, res_reduce=4):
#     B, C, H, W = img.shape
#     batch = AttrDict()
#     batch.xc, batch.yc = get_subscaled_img_data(img, W//res_reduce)

#     num_pixels = H*W
#     img = img.view(B, C, -1)

#     num_tar = num_pixels if target_all else \
#             torch.randint(low=3, high=max_num_points-3, size=[1]).item()
#     idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_tar].to(img.device)
#     x1, x2 = idxs//W, idxs%W
#     batch.xt = torch.stack([
#         2*x1.float()/(H-1) - 1,
#         2*x2.float()/(W-1) - 1], -1).to(img.device)
#     batch.yt = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
#             .transpose(-2, -1) - 0.5).to(img.device)

#     batch.x = torch.cat((batch.xc, batch.xt), dim=1)
#     batch.y = torch.cat((batch.yc, batch.yt), dim=1)

#     return batch

def coord_to_img(x, y, shape):
    x = x.cpu()
    y = y.cpu()
    B = x.shape[0]
    C, H, W = shape

    I = torch.zeros(B, 3, H, W)
    I[:,0,:,:] = 0.61
    I[:,1,:,:] = 0.55
    I[:,2,:,:] = 0.71

    x1, x2 = x[...,0], x[...,1]
    x1 = ((x1+1)*(H-1)/2).round().long()
    x2 = ((x2+1)*(W-1)/2).round().long()
    for b in range(B):
        for c in range(3):
            I[b,c,x1[b],x2[b]] = y[b,:,min(c,C-1)]

    return I

def task_to_img(xc, yc, xt, yt, shape):
    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    task_img = torch.zeros(B, 3, H, W).to(xc.device)
    task_img[:,2,:,:] = 1.0
    task_img[:,1,:,:] = 0.4
    for b in range(B):
        for c in range(3):
            task_img[b,c,xc1[b],xc2[b]] = yc[b,:,min(c,C-1)] + 0.5
    task_img = task_img.clamp(0, 1)

    completed_img = task_img.clone()
    for b in range(B):
        for c in range(3):
            completed_img[b,c,xt1[b],xt2[b]] = yt[b,:,min(c,C-1)] + 0.5
    completed_img = completed_img.clamp(0, 1)

    return task_img, completed_img


def img_to_task_pixelate(img, num_ctx=None,
        max_num_points=None, target_all=False, level=None):
    B, C, H, W = img.shape
    c = [0.95, 0.9, 0.85, 0.75, 0.65][level - 1]
    
    img_subscale = torchvision.transforms.functional.resize(img, size=(int(W * c),int(W * c)), interpolation=InterpolationMode.NEAREST)
    img_upscale = torchvision.transforms.functional.resize(img_subscale, size=(W,W), interpolation=InterpolationMode.NEAREST)
    
    num_pixels = H*W
    img = img.view(B, C, -1)
    img_upscale = img_upscale.view(B, C, -1)

    batch = AttrDict()
    max_num_points = max_num_points or num_pixels
    num_ctx = num_ctx or \
            torch.randint(low=3, high=max_num_points-3, size=[1]).item()
    num_tar = max_num_points - num_ctx if target_all else \
            torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()
    num_points = num_ctx + num_tar
    idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(img.device)
    
    y_all = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(img.device)
    y_all_rescale = (torch.gather(img_upscale, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(img.device)
    
    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]

    batch.yc = y_all_rescale[:,:num_ctx]
    batch.yt = y_all[:,num_ctx:]
    batch.y = torch.cat((batch.yc, batch.yt), dim=1)

    return batch


# Noise corruption taken from benchmark (with minor mods):
# - https://github.com/hendrycks/robustness/blob/305054464935cdf4c5f182f1387a1cc506854d49/ImageNet-C/create_c/make_cifar_c.py
def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    # x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), -0.5, 0.5)# * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    # x = np.array(x) / 255.
    x = x + 0.5 # range (0,1)
    x_pert = np.clip(np.random.poisson(x * c) / c, 0, 1) #* 255
    x_pert = x_pert - 0.5
    return x_pert


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    # x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = sk.util.random_noise(x+0.5, mode='s&p', amount=c)
    x = x - 0.5
    return np.clip(x, -0.5, 0.5) #* 255


def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    # x = np.array(x) / 255.
    x = x + 0.5 # range (0,1)
    x_pert =  np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1)# * 255
    x_pert = x_pert - 0.5
    return x_pert
