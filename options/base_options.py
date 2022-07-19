import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='tmp', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='tmp', help='models are saved here')
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--dataroot', type=str, default='tmp',help='path of datasets, see readme for data structure')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--dimension', default='3D', type=str, help='# 2D or 3D PTNet')

        # dataloading:
        self.parser.add_argument('--extension', type=str, default='.nii.gz', help='file extension (*.nii/*.nii.gz(')
        self.parser.add_argument('--norm_perc', type=float, default=99.95, help='intensity over norm_perc% percentile will be set as 1 while performing 0-1 normalization')
        self.parser.add_argument('--patch_size', type=tuple, default=(64, 64, 64),
                                 help='size of 3D patch')
        self.parser.add_argument('--remove_bg', type=bool, default=False, help='whether to only sample inside the scan (non-zero), requires to have positive value for any foreground pixels')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display size during training in ./web/index.html')


        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        assert self.opt.dimension in ['2D','3D','2d','3d'], 'dimension is not supported'

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save :
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
