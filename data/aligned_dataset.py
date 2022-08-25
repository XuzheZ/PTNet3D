### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### This script was modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from data.base_dataset import BaseDataset
from data_util import *
import torch
import nibabel as nib
import numpy as np
import random


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ###
        ### input A (source domain)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.extension))
        assert self.A_paths, 'modality A can not find files with extension ' + opt.extension
        ### input B (target domain)
        ### if you are converting T1w to T2w, please put training T1w scans into train_A and training T2w scans into train_B
        dir_B = '_B'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B, opt.extension))
        assert self.B_paths, 'modality B can not find files with extension ' + opt.extension

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        x, y, z = self.opt.patch_size

        ### modality A

        tmp_scansA = np.squeeze(nib.load(self.A_paths[index]).get_fdata())
        tmp_scansB = np.squeeze(nib.load(self.B_paths[index]).get_fdata())
        assert tmp_scansA.shape == tmp_scansB.shape, 'paired scans must have the same shape'

        tmp_scansA[tmp_scansA < 0] = 0
        tmp_scansB[tmp_scansB < 0] = 0
        tmp_scansA = norm_img(tmp_scansA, self.opt.norm_perc)
        tmp_scansB = norm_img(tmp_scansB, self.opt.norm_perc)

        tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        tmp_scansB = torch.unsqueeze(torch.from_numpy(tmp_scansB), 0)
        _, x1, y1, z1 = tmp_scansA.shape

        if self.opt.dimension.startswith('2'):
            if self.opt.remove_bg:
                bound = get_bounds(tmp_scansA)
            else:
                bound = [0, x1, 0, y1, 0, z1]

            slice_idx = random.sample(range(bound[-2], bound[-1]), 1)[0]

            input_dict = {'img_A': tmp_scansA[:, :, :, slice_idx],
                          'img_B': tmp_scansB[:, :, :, slice_idx]}

        elif self.opt.dimension.startswith('3'):
            for i in range(3):
                assert tmp_scansA.shape[i] >= self.opt.patch_size[i], self.A_paths[index] + ' ' + str(
                    i + 1) + ' dimension is smaller than corresponding patch size'

            if self.opt.remove_bg:
                bound = get_bounds(tmp_scansA[0])
                assert bound[1] - x > bound[0], 'first dimension is smaller than patch size after removing background, ' \
                                                'cosider padding or setting remove_bg as false '
                assert bound[3] - y > bound[
                    2], 'second dimension is smaller than patch size after removing background, ' \
                        'cosider padding or setting remove_bg as false '
                assert bound[5] - z > bound[4], 'third dimension is smaller than patch size after removing background, ' \
                                                'cosider padding or setting remove_bg as false '

                x_idx = random.sample(range(bound[0], bound[1] - x), 1)[0]
                y_idx = random.sample(range(bound[2], bound[3] - y), 1)[0]
                z_idx = random.sample(range(bound[4], bound[5] - z), 1)[0]
            else:
                bound = [0, x1, 0, y1, 0, z1]
                if x1 - x == 0:
                    x_idx = 0
                else:
                    x_idx = random.sample(range(bound[0], bound[1] - x), 1)[0]
                if y1 - y == 0:
                    y_idx = 0
                else:
                    y_idx = random.sample(range(bound[2], bound[3] - y), 1)[0]
                if z1 - z == 0:
                    z_idx = 0
                else:
                    z_idx = random.sample(range(bound[4], bound[5] - z), 1)[0]
            input_dict = {'img_A': tmp_scansA[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                          'img_B': tmp_scansB[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Paired/Aligned Dataset'
