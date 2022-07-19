import os
from options.test_options import TestOptions
from models.models import create_model

import nibabel as nib
import numpy as np
import torch
from data.data_util import norm_img, patch_slicer, get_bounds

opt = TestOptions().parse(save=False)

PTNet, _, _ = create_model(opt)
PTNet.cuda()
PTNet.eval()
PTNet.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, opt.whichmodel)))

test_path = os.path.join(opt.dataroot, 'T1s')

des = os.path.join(opt.dataroot, opt.name+'_'+opt.whichmodel)
if not os.path.exists(des):
    os.mkdir(des)

test_lst = [i for i in os.listdir(test_path) if i.endswith(opt.extension)]

with torch.no_grad():
    for i in test_lst:
        # load test scans, define output name
        print('Loading test image: ' + i)
        pred_name = i.split(opt.extension)[0] + '_PTNetSynth' + opt.extension
        try:
            nib.load(os.path.join(test_path,i))
        except ValueError:
            nib.Nifti1Header.quaternion_threshold = -1e-06
        tmp_scans = np.squeeze(nib.load(os.path.join(test_path,i)).get_fdata())
        tmp_scans[tmp_scans < 0] = 0

        # define matrix to store prediction and normalization matrices
        pred = np.zeros(tmp_scans.shape)
        tmp_norm = np.zeros(tmp_scans.shape)
        # normalize image
        tmp_scans = norm_img(tmp_scans, opt.norm_perc)
        if opt.dimension.startswith('3'):
            scan_patches, tmp_path, tmp_idx = patch_slicer(tmp_scans, tmp_scans, opt.patch_size,  tuple(dim//2 for dim in opt.patch_size),
                                                           remove_bg=opt.remove_bg, test=True, ori_path=None)
            # go through all patches
            for idx, patch in enumerate(scan_patches):
                ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
                ipt = ipt.reshape((1,1,)+ipt.shape)
                tmp_pred = PTNet(ipt)
                patch_idx = tmp_idx[idx]
                patch_idx = (slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
                pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
                tmp_norm[patch_idx] += 1

            pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
        elif opt.dimension.startswith('2'):
            if opt.remove_bg:
                bound = get_bounds(tmp_scans)
            else:
                x1, y1, z1 = tmp_scans.shape
                bound = [0, x1, 0, y1, 0, z1]
            for idx in range(bound[-2],bound[-1]):
                ipt = torch.from_numpy(tmp_scans[:,:,idx]).to(dtype=torch.float).cuda()
                ipt = ipt.reshape((1, 1,) + ipt.shape)
                pred[idx] = torch.squeeze(PTNet(ipt)).detach().cpu().numpy()
        ori_scan = nib.load(os.path.join(test_path, i))
        sav_img = nib.Nifti1Image(pred, ori_scan.affine, header=ori_scan.header)
        nib.save(sav_img, os.path.join(des, pred_name))
