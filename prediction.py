from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
import models
import torch.nn.functional as F
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='DispNetS')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--colored', default=1, help='save colored or save for benchmark submission')
parser.add_argument('--eval', type=bool, default=False, help='Evaluate quality')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
dispNet = models.DispNetS()
dispNet = nn.DataParallel(dispNet)
dispNet.cuda()


# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
#dispNet.module.load_state_dict(state_dict['state_dict'])
dispNet.load_state_dict(state_dict['state_dict'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)

    if args.eval == True:
        avg_test_scalars = AverageMeterDict()

    for batch_idx, sample in enumerate(TestImgLoader):

        if args.eval == True:
            loss, scalar_outputs, disp_est_tn = test_sample(sample, True)                    
            print('loss = {:.4f}, EPE = {:.4f}, D1 = {:.4f}, Th1 = {:.4f}, Th2 = {:.4f}, Th3 = {:.4f}'.format(loss,  
                                                                                scalar_outputs['EPE'][0], scalar_outputs['D1'][0],
                                                                                scalar_outputs['Thres1'][0], scalar_outputs['Thres2'][0],
                                                                                scalar_outputs['Thres3'][0]))

            #loss, EPE, D1, Th1, Th2, Th3, disp_est_tn = test_sample(sample, True)
            #print('loss = {:.4f}, EPE = {:.4f}, D1 = {:.4f}, Th1 = {:.4f}, Th2 = {:.4f}, Th3 = {:.4f}'.format(loss, EPE[0], D1[0], Th1[0], Th2[0], Th3[0]))
            avg_test_scalars.update(scalar_outputs)    
            del scalar_outputs
        else:
            disp_est_tn = test_sample(sample)
    
        disp_est_np = tensor2numpy(disp_est_tn)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            assert len(disp_est.shape) == 2

            #disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            disp_est = np.array(disp_est[:, :], dtype=np.float32)
            name = fn.split('/')
            fn = os.path.join("predictions", '_'.join(name[2:]))
            print(fn)

            if float(args.colored) == 1:            
                disp_est = kitti_colormap(disp_est)
                cv2.imwrite(fn, disp_est)
            else:
                disp_est = np.round(disp_est * 256).astype(np.uint16)
                io.imsave(fn, disp_est)
    
    if args.eval == True:
        avg_test_scalars = avg_test_scalars.mean()
        print("avg_test_scalars", avg_test_scalars)

    print("Done!")


@make_nograd_func
def test_sample(sample, compute_metrics=False):
    if compute_metrics==True:
        dispNet.eval()

        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()

        disp_ests = dispNet(imgL, imgR)
        disp_ests[-1] = torch.squeeze(disp_ests[-1], 1)
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        loss = model_loss(disp_ests, disp_gt, mask)

        scalar_outputs = {"loss": loss}
        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
                
        return tensor2float(loss), tensor2float(scalar_outputs), disp_ests[-1]
        
    else:
        dispNet.eval()
        disp_ests = dispNet(sample['left'].cuda(), sample['right'].cuda())
        disp_ests[-1] = torch.squeeze(disp_ests[-1], 1)
        return disp_ests[-1]

def model_loss(disp_ests, disp_gt, mask):
    weights = [1, 0.7, 0.5, 0.25]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)


if __name__ == '__main__':
    test(args)
