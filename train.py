from __future__ import print_function, division
import os
import gc
import time
import argparse
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
import models
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datasets import __datasets__
from utils import *

import cv2

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='DispNetS')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=200, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)


# create summary logger
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
dispNet = models.DispNetS()
dispNet = nn.DataParallel(dispNet)
dispNet.cuda()

optimizer = optim.Adam(dispNet.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0

if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("Loading the latest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    dispNet.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("Loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    
    # # DispNetS from SfmLearner has one input, so check if conv1.0.weight has 3 channel.
    # # If so, increae its channels to 6
    # conv10w = state_dict['state_dict']['conv1.0.weight']
    # if conv10w.size()[1] == 3:
    #     state_dict['state_dict']['conv1.0.weight'] = torch.cat([conv10w, conv10w], 1)

    dispNet.load_state_dict(state_dict['state_dict'], strict=False)
#else:
#    dispNet.init_weights()


print("Start at epoch {}".format(start_epoch))

# DataParallel after loading pre-trained model due to init_weight()
# - 'DataParallel' object has no attribute 'init_weights'
#dispNet = nn.DataParallel(dispNet)

def train():
    best_checkpoint_loss = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'state_dict': dispNet.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()

        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)

        # saving new best checkpoint
        if avg_test_scalars['loss'] < best_checkpoint_loss:
            best_checkpoint_loss = avg_test_scalars['loss']
            print("Overwriting best checkpoint")
            checkpoint_data = {'epoch': epoch_idx, 'state_dict': dispNet.module.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/best.ckpt".format(args.logdir))

        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    dispNet.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']

    #gt_ = disp_gt[0, :, :]
    #gt_ = torch.squeeze(gt_)
    #gt_ = tensor2numpy(gt_)
    #gt_ = np.array(gt_[:, :], dtype=np.uint8)

    #fn = os.path.join("predictions", "groundtruth.jpg")
    #io.imsave(fn, gt_)
    

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = dispNet(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    #print("****")
    #print(len(disp_ests))
    #print(disp_ests[0].size())
    #print(disp_gt.size())
    #print(mask.size())    

    #disp_gt2 = F.avg_pool2d(disp_gt, 5, stride = 2, padding=2)
    #mask2 = (disp_gt2 < args.maxdisp) & (disp_gt2 > 0)    
    #disp_gt3 = F.avg_pool2d(disp_gt2, 5, stride = 2, padding=2)
    #mask3 = (disp_gt3 < args.maxdisp) & (disp_gt3 > 0)
    #disp_gt4 = F.avg_pool2d(disp_gt3, 5, stride = 2, padding=2)
    #mask4 = (disp_gt4 < args.maxdisp) & (disp_gt4 > 0)

    #disp_gts = (disp_gt, disp_gt2, disp_gt3, disp_gt4)
    #masks = (mask, mask2, mask3, mask4)
    #loss = model_loss(disp_ests, disp_gt, mask)    

    disp_ests[0] = torch.squeeze(disp_ests[0], 1)
    for i in range(1, 4):        
        disp_ests[i] = F.interpolate(disp_ests[i], scale_factor=2**i, mode='bilinear')
        disp_ests[i] = torch.squeeze(disp_ests[i], 1)

    #gt_ = disp_ests[0][0, :, :]
    #gt_ = tensor2numpy(gt_)
    #gt_ = np.array(gt_[:, :], dtype=np.float32)
    #gt_ = np.round(gt_ * 256).astype(np.uint16)

    #fn = os.path.join("predictions", "estimate.jpg")
    #io.imsave(fn, gt_)
    
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
            #image_outputs["errormap"] = [disp_error_image_func(disp_ests[0], disp_gt)]
            #scalar_outputs["EPE"] = [EPE_metric(disp_ests[0], disp_gt, mask)]
            #scalar_outputs["D1"] = [D1_metric(disp_ests[0], disp_gt, mask)]
            #scalar_outputs["Thres1"] = [Thres_metric(disp_ests[0], disp_gt, mask, 1.0)]
            #scalar_outputs["Thres2"] = [Thres_metric(disp_ests[0], disp_gt, mask, 2.0)]
            #scalar_outputs["Thres3"] = [Thres_metric(disp_ests[0], disp_gt, mask, 3.0)]            

    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    dispNet.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = dispNet(imgL, imgR)
    disp_ests[0] = torch.squeeze(disp_ests[0], 1)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def model_loss(disp_ests, disp_gt, mask):
    weights = [1, 0.7, 0.5, 0.25]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    #for disp_est, disp_gt, mask, weight in zip(disp_ests, disp_gts, masks, weights):        
    #    all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

def model_loss_test(disp_ests, disp_gt, mask):
    weights = [1.0]
    all_losses = []
    
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)


if __name__ == '__main__':
    train()
