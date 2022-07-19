### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import math
from models.models import create_model
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
from util.image_pool import ImagePool
from models.networks import GANLoss, feature_loss, discriminate
def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0

##############################################################################
# Initialize options
##############################################################################

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
start_epoch, epoch_iter = 1, 0
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
ler = opt.lr

##############################################################################
# Initialize dataloader
##############################################################################

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

##############################################################################
# Initialize networks
##############################################################################

PTNet, D, ext_discriminator = create_model(opt)
PTNet.cuda()
D.cuda()
ext_discriminator.cuda()

##############################################################################
# Initialize util components
##############################################################################

optimizer_PTNet = torch.optim.Adam(PTNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
fake_pool = ImagePool(0)

CE = nn.CrossEntropyLoss()
criterionGAN = GANLoss(use_lsgan=not False, tensor=torch.cuda.FloatTensor)
mse = torch.nn.MSELoss()

# training/display parameter
visualizer = Visualizer(opt)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

##############################################################################
# Training code
##############################################################################

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ##############################################################################
        # Forward Pass
        ##############################################################################

        input_image = Variable(data['img_A'].cuda())
        target_image = Variable(data['img_B'].cuda())

        # Synthesize and MSE loss
        generated = PTNet(input_image)
        loss_mse = mse(generated, target_image)

        # Fake Detection and Loss
        pred_fake_pool = discriminate(D, fake_pool, input_image, generated, use_pool=True)
        loss_D_fake = criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = discriminate(D, fake_pool, input_image, target_image)
        loss_D_real = criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = D.forward(torch.cat((input_image, generated), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat, loss_G_GAN_Feat_ext = feature_loss(opt, target_image, generated, pred_real, pred_fake,
                                                            ext_discriminator)

        # Compute overall loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = loss_mse * 100.0 + loss_G_GAN + loss_G_GAN_Feat_ext * 10.0 + loss_G_GAN_Feat * 10.0
        loss_dict = dict(
            zip(['MSE', 'G_GAN', 'G_GAN_Feat_ext', 'G_GAN_Feat', 'D_fake', 'D_real'], [loss_mse.item(),
                                                                                       loss_G_GAN.item(),
                                                                                       loss_G_GAN_Feat_ext.item(),
                                                                                       loss_G_GAN_Feat.item(),
                                                                                       loss_D_fake.item(),
                                                                                       loss_D_real.item()]), )
        ##############################################################################
        # Backward
        ##############################################################################

        # update generator weights
        optimizer_PTNet.zero_grad()
        loss_G.backward()
        optimizer_PTNet.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ##############################################################################
        # Display results, print out loss, and save latest model
        ##############################################################################

        # print out loss
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        # display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0, :, :, :, 15], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0, :, :, :, 15])),
                                   ('real_image', util.tensor2im(data['image'][0, :, :, :, 15]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_latest.pth'))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_ckpt%d%d.pth' % (epoch, total_steps)))

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        for param_group in optimizer_PTNet.param_groups:
            param_group['lr'] = ler
            print('change lr to ')
            print(param_group['lr'])
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = ler
