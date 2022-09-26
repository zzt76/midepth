from math import ldexp
from sched import scheduler
import matplotlib
import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
# import wandb
from tqdm import tqdm
from datetime import datetime

import model_io
# from models.unet import UNet
from models.unet import UNet
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss
from utils import RunningAverage, colorize

from tensorboardX import SummaryWriter

PROJECT = "MIDepth"
logging = True


def is_rank_zero(args):
    return args.rank == 0


def log_images(writer, img, depth, pred, args, step):
    img = torchvision.utils.make_grid(img.detach())
    depth = torchvision.utils.make_grid(depth / args.max_depth)
    pred = torchvision.utils.make_grid(pred / args.max_depth)
    writer.add_image("Train/img", img, step)
    writer.add_image("Train/depth", depth, step)
    writer.add_image("Train/pred", pred, step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    model = UNet(cout=1)
    optimizer_state_dict = None
    scheduler_state_dict = None
    epoch = 0
    if(args.resume_path != ""):
        print(f"Loading from {args.resume_path}.")
        if(args.finetune):
            model, optimizer_state_dict, scheduler_state_dict, epoch = model_io.load_checkpoint_finetune(args.resume_path, model)
        else:
            model, optimizer_state_dict, scheduler_state_dict, epoch = model_io.load_checkpoint(args.resume_path, model)
    # model(torch.rand([1, 3, args.input_height, args.input_width]))  # initialize model with dummy values

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print("Gpu:", args.gpu, "rank:", args.rank, "batch_size:", args.batch_size, "workers:", args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=False)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = epoch
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=optimizer_state_dict, scheduler_state_dict=scheduler_state_dict)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None, scheduler_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        print(f"Training {experiment_name}")
        writer = SummaryWriter("logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    run_id = f"{dt.now().strftime('%h-%d_%H-%M')}-node_bs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    # l1_loss = nn.L1Loss()
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    # if args.same_lr:
    #     print("Using same LR")
    #     params = model.parameters()
    # else:
    #     print("Using diff LR")
    #     m = model.module if args.multigpu else model
    #     params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
    #               {"params": m.get_10x_lr_params(), "lr": lr}]
    params = model.parameters()
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if args.resume_path != "" and optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if args.resume_path != "" and scheduler_state_dict is not None:
        # scheduler.step(args.epoch + 1)
        scheduler.load_state_dict(scheduler_state_dict)
    ################################################################################################

    # max_iter = len(train_loader) * epochs
    cumulated_loss = 10
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            # assert(torch.isnan(img))
            pred = model(img) * args.max_depth
            # assert(torch.isnan(pred))
            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            # l_l1 = l1_loss(pred[mask], depth[mask])

            loss = l_dense
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()

            cumulated_loss += loss.item()
            if should_log and step % args.log_loss_freq == 0:
                writer.add_scalar('info/lr', scheduler.get_last_lr(), step)
                cumulated_loss = cumulated_loss / args.log_loss_freq
                writer.add_scalar(f'info/last_{args.log_loss_freq}_loss', cumulated_loss, step)
                cumulated_loss = 0

            if should_log and step % args.log_image_freq == 0:
                img_log = img.detach().cpu()
                depth_log = depth.detach().cpu()
                pred_log = pred.detach().cpu()
                log_images(writer, img_log, depth_log, pred_log, args, step)

            step += 1
            # scheduler.step()

            ########################################################################################################

            if should_write and epoch >= args.online_eval_after and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    writer.add_scalar(f'OnlineEval/{l_dense.name}', val_si.get_value(), step)
                    for k, v in metrics.items():
                        writer.add_scalar(f"OnlineEval/{k}", v, step)

                    model_io.save_checkpoint(model, optimizer, scheduler, epoch, f"{experiment_name}_{run_id}_latest.pt", root=os.path.join(root, "checkpoints"))

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, scheduler, epoch, f"{experiment_name}_{run_id}_best.pt", root=os.path.join(root, "checkpoints"))
                    print(f"Best checkpoint {experiment_name}_{epoch}_{step} saved.")
                    best_loss = metrics['abs_rel']
                model.train()
            #################################################################################################

        scheduler.step()
        if should_write:
            model_io.save_checkpoint(model, optimizer, scheduler, epoch, f"{experiment_name}_{epoch}.pt",
                                     root=os.path.join(root, "checkpoints"))
            print(f"Checkpoint {experiment_name}_{epoch}.pt saved.")

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            pred = model(img)*args.max_depth_eval

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            val_si.append(l_dense.item())

            # pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                              int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                  int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            # valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--log_loss-freq', '--log_loss_freq', default=50, type=int, help='log image frequency')
    parser.add_argument('--log_image-freq', '--log_image_freq', default=500, type=int, help='log image frequency')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume_path", default="", type=str, help="Resume from checkpoint path")
    parser.add_argument('--finetune', default=False, help="If set, resume from checkpoint but not load optimizer and scheduler", action='store_true')
    parser.add_argument("--online_eval_after", default=20, type=int, help="Resume from checkpoint path")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print("Current rank is", args.rank)
        # port = np.random.randint(15000, 15025)
        port = 15025
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print("Distributed url is", args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
