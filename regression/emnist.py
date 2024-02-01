import os
import os.path as osp
import argparse
import yaml
import torch
import numpy as np
import time
import uncertainty_toolbox as uct
from attrdict import AttrDict
from tqdm import tqdm
from copy import deepcopy
from PIL import Image

from data.image import img_to_task, img_to_task_pixelate, task_to_img
from data.emnist import EMNIST
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

# torch.set_default_dtype(torch.float64)
# torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', choices=['train', 'eval', 'eval_all_metrics', 'plot', 'plot_samples'], default='train')
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # Data
    parser.add_argument('--max_num_points', type=int, default=200) # TODO: could change to half of num_pixels=392
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])

    # Model
    parser.add_argument('--model', type=str, default="bayesnp")

    # Specific to BayesNP, BayesCNP
    parser.add_argument('--max_num_cavi_steps', type=int, default=1)
    parser.add_argument('--min_num_cavi_steps', type=int, default=-1)
    # Specific to BayesNPMoG
    parser.add_argument('--n_components', type=int, default=3)

    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=5)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)
    parser.add_argument('--eval_pixelate', action='store_true', default=False) # NEW
    parser.add_argument('--eval_pixelate_lvl', type=int, default=1) # NEW
    parser.add_argument('--eval_num_ctx', type=int, default=None) # NEW

    # Plot
    parser.add_argument('--plot_seed', type=int, default=1)
    parser.add_argument('--plot_num_imgs', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_bs', type=int, default=50)
    parser.add_argument('--plot_num_ctx', type=int, default=100)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=int, default=None)
    parser.add_argument('--plot_noise_type', type=int, default=None)

    args = parser.parse_args()

    if args.expid is None:
        if (args.mode == 'train') & (not args.resume):
            args.expid = '{}_s{}'.format(args.train_seed, args.max_num_cavi_steps)
            if args.model in ('bayesnpmog','bayescnpmog'):
                args.expid += '_k{}'.format(args.n_components)
        else:
            args.expid = 'default'

    if args.expid is not None:
        args.root = osp.join(results_path, 'emnist', args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'emnist', args.model)

    if args.min_num_cavi_steps == -1:
        args.min_num_cavi_steps = args.max_num_cavi_steps

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/emnist/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if args.pretrain:
        assert args.model == 'tnpa'
        config['pretrain'] = args.pretrain

    # if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd"]:

    # Hack to make emnist.py work with MixtureOfDiagNormals
    args.is_fp64 = False # TODO: remove if no longer used
    if args.model in ('bayesnpmog','bayescnpmog'):
        config['n_components'] = args.n_components
        # torch.set_default_dtype(torch.float64)
        # args.is_fp64 = True

    model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'eval_all_metrics':
        eval_all_metrics(args, model)
    elif args.mode == 'plot':
        plot(args, model)
    elif args.mode == 'plot_samples':
        plot_samples(args, model)

# NOTE: train_seed not used (whether can seed data_loader?)
def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = EMNIST(train=True, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.cuda()
            if args.is_fp64:
                x = x.double()
            batch = img_to_task(x,
                max_num_points=args.max_num_points)
            optimizer.zero_grad()

            if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "npsa", "bayesnpmog"]:
                outs = model(batch, num_samples=args.train_num_samples)
            elif args.model in ["bayesnp", "bayescnp"]:
                outs = model(batch, num_samples=args.train_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
            else:
                outs = model(batch)

            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    # regular eval
    args.mode = 'eval'
    eval(args, model)
    # unseen eval
    args.class_range = [10,47]
    eval(args, model)
    args.class_range = [0,10] # reset class range
    # # noise
    # args.t_noise = -1
    # eval(args, model)
    # args.t_noise = None # reset noise
    # # super-res 2
    # args.eval_pixelate = True
    # args.eval_pixelate_lvl = 2
    # eval(args, model)
    # # super-res 4
    # args.eval_pixelate_lvl = 4
    # eval(args, model)

def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=0)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        if args.eval_pixelate:
            batches.append(img_to_task_pixelate(
                x, num_ctx=args.eval_num_ctx, max_num_points=args.max_num_points,
                level=args.eval_pixelate_lvl)
            )
        else:
            batches.append(img_to_task(
                x, num_ctx=args.eval_num_ctx, max_num_points=args.max_num_points,
                t_noise=args.t_noise)
            )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'emnist')
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f'{c1}-{c2}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    if args.eval_pixelate:
        filename += f'_sr{args.eval_pixelate_lvl}'
    if args.eval_num_ctx is not None:
        filename += f'_c{args.eval_num_ctx}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))


def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f'eval_{c1}-{c2}'
            if args.t_noise is not None:
                eval_logfile += f'_s{args.max_num_cavi_steps}_{args.t_noise}'
            if args.eval_pixelate:
                eval_logfile += f'_s{args.max_num_cavi_steps}_sr{args.eval_pixelate_lvl}'
            if args.eval_num_ctx is not None:
                eval_logfile += f'_c{args.eval_num_ctx}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    if args.eval_pixelate:
        filename += f'_sr{args.eval_pixelate_lvl}'
    if args.eval_num_ctx is not None:
        filename += f'_c{args.eval_num_ctx}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage(is_std=True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
                if args.is_fp64:
                    batch[key] = val.cuda().double()
            
            if args.model in ["np", "anp", "bnp", "banp", "npsa", "bayesnpmog"]:
                outs = model(batch, args.eval_num_samples)
            elif args.model in ["bayesnp", "bayescnp"]:
                outs = model(batch, args.eval_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    c1, c2 = args.class_range
    line = f'{args.model}:{args.expid} {c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    if args.eval_pixelate:
        line += f'_sr{args.eval_pixelate_lvl} '
    if args.eval_num_ctx is not None:
        line += f'_c{args.eval_num_ctx} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


def eval_all_metrics(args, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
    model.load_state_dict(ckpt.model)

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    if not osp.isdir(path):
        os.makedirs(path)
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(3)] # 3 types of metrics
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
                if args.is_fp64:
                    batch[key] = val.cuda().double()

            if args.model in ["np", "anp", "bnp", "banp", "npsa", "bayesnpmog"]:
                outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
                ll = model(batch, num_samples=args.eval_num_samples)
            elif args.model in ["tnpa", "tnpnd"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt,
                    num_samples=args.eval_num_samples
                )
                ll = model(batch)
            elif args.model in ["bayesnp", "bayescnp"]:
                outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
                ll = model(batch, num_samples=args.eval_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
            else:
                outs = model.predict(batch.xc, batch.yc, batch.xt)
                ll = model(batch)

            mean, std = outs.loc, outs.scale

            # shape: (num_samples, 1, num_points, 1)
            if mean.dim() == 4:
                var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                std = var.sqrt().squeeze(0)
                mean = mean.mean(dim=0).squeeze(0)
            
            mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {'tar_ll': ll.tar_ll.item()}

            batch_metrics = [acc, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid}:{c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    
    line += '\n'

    for ravg in ravgs:
        line += ravg.info()
        line += '\n'

    filename = f'eval_{c1}-{c2}_all_metrics'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, args.expid, filename), mode='w')
    logger.info(line)

    return line


def plot(args, model):
    if args.mode == 'plot':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).cuda()
    batch = img_to_task(test_data, max_num_points=None, num_ctx=args.plot_num_ctx, target_all=True, t_noise=args.t_noise, plot_noise_type=args.plot_noise_type)
    
    if args.is_fp64:
        for key, val in batch.items():
            batch[key] = val.double()

    model.eval()
    with torch.no_grad():
        if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd", "npsa", "bayesnpmog"]:
            outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
        elif args.model in ["bayesnp", "bayescnp"]:
            outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
        else:
            outs = model.predict(batch.xc, batch.yc, batch.xt)

    mean = outs.mean
    # shape: (num_samples, 1, num_points, 1)
    if mean.dim() == 4:
        mean = mean.mean(dim=0)

    task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, mean, shape=(1,28,28))
    _, orig_img = task_to_img(batch.xc, batch.yc, batch.xt, batch.yt, shape=(1,28,28))

    task_img = (task_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    orig_img = (orig_img * 255).int().cpu().numpy().transpose(0,2,3,1)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(args.plot_num_imgs):
        # Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
        # Image.fromarray(task_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task.jpg' % (i+1))
        # Image.fromarray(completed_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed.jpg' % (i+1))
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((28,28),Image.NEAREST).save(save_dir + '/%d_orig.jpg' % (i+1))
        Image.fromarray(task_img[i].astype(np.uint8)).resize((28,28),Image.NEAREST).save(save_dir + '/%d_task.jpg' % (i+1))
        Image.fromarray(completed_img[i].astype(np.uint8)).resize((28,28),Image.NEAREST).save(save_dir + '/%d_completed.jpg' % (i+1))

def plot_samples(args, model):
    if args.mode == 'plot_samples':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).cuda()
    
    list_num_ctx = [10, 20, 50, 100, 150]
    batches = [img_to_task(test_data, max_num_points=None, num_ctx=i, target_all=True) for i in list_num_ctx]
    all_samples = []

    if args.is_fp64:
        for key, val in batch.items():
            batch[key] = val.double()
    
    model.eval()
    with torch.no_grad():
        for batch in batches:
            if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd", "npsa", "bayesnpmog"]:
                samples = model.sample(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
            elif args.model in ["bayesnp", "bayescnp"]:
                samples = model.sample(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples, max_n_steps=args.max_num_cavi_steps, min_n_steps=args.min_num_cavi_steps)
            else:
                samples = model.sample(batch.xc, batch.yc, batch.xt)
            all_samples.append(samples)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'sample_plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    # save original images
    _, orig_img = task_to_img(batches[-1].xc, batches[-1].yc, batches[-1].xt, batches[-1].yt, shape=(1,28,28)) # (num_imgs, 32, 32, 3)
    orig_img = (orig_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    for i in range(args.plot_num_imgs):
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
    
    for i in range(len(list_num_ctx)):
        num_ctx = list_num_ctx[i]
        batch = batches[i]
        samples = all_samples[i]

        for j in range(args.eval_num_samples):
            task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, samples[j], shape=(1,28,28)) # (num_imgs, 32, 32, 3)

            task_img = (task_img * 255).int().cpu().numpy().transpose(0,2,3,1)
            completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)

            for k in range(args.plot_num_imgs):
                Image.fromarray(task_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task_%d_ctx.jpg' % (k+1, num_ctx))
                Image.fromarray(completed_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed_%d_ctx_%d_samples.jpg' % (k+1, num_ctx, j+1))

if __name__ == '__main__':
    main()
