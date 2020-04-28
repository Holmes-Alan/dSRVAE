from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from modules import VAE_denoise_vali, discriminator, VAE_SR, VGGFeatureExtractor
from data import get_training_set
import pdb
import socket
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--pretrained_iter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/data/NTIRE2020')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped LR image')
parser.add_argument('--pretrained_sr', default='VAE_good_v5.pth', help='sr pretrained base model')
parser.add_argument('--pretrained_D', default='GAN_discriminator_110.pth', help='sr pretrained base model')
parser.add_argument('--model_type', default='GAN', help='model name')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--pretrain_flag', default=False, help='pretrain generator')
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='logs/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)



def train(epoch):
    G_epoch_loss = 0
    D_epoch_loss = 0
    adv_epoch_loss = 0
    vgg_epoch_loss = 0
    recon_epoch_loss = 0
    G.train()
    D.train()
    feat_extractor.eval()
    denoiser.eval()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, ref = batch[0], batch[1], batch[2]
        minibatch = input.size()[0]
        real_label = torch.ones(minibatch)
        fake_label = torch.zeros(minibatch)

        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            ref = Variable(ref).cuda(gpus_list[0])
            real_label = Variable(real_label).cuda(gpus_list[0])
            fake_label = Variable(fake_label).cuda(gpus_list[0])

        down = torch.nn.Upsample(scale_factor=0.25, mode='bicubic')
        up = torch.nn.Upsample(scale_factor=4, mode='bicubic')

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = False


        G_optimizer.zero_grad()
        down_ref = down(ref)
        with torch.no_grad():
            std_z = torch.from_numpy(np.random.normal(0, 1, (input.shape[0], 512))).float()
            z = Variable(std_z, requires_grad=False).cuda(gpus_list[0])
            Denoise_LR = denoiser(input, z)

        SR = G(Denoise_LR)
        SR_tar = G(target)

        SR_feat = feat_extractor(SR).detach()
        SR_tar_feat = feat_extractor(SR_tar).detach()
        Tar_feat = feat_extractor(target).detach()


        D_fake_decision_1 = D(SR)
        D_fake_decision_2 = D(SR_tar)
        D_real_decision = D(ref).detach()

        GAN_loss = (BCE_loss(D_fake_decision_1, real_label)
                    + BCE_loss(D_fake_decision_2, real_label)
                    + BCE_loss(D_real_decision, fake_label)) / 3.0


        recon_loss = (L1_loss(down(SR), target) + L1_loss(SR_tar, SR)) / 2.0
        vgg_loss = (L1_loss(down(SR_feat), Tar_feat) + L1_loss(SR_tar_feat, SR_feat)) / 2.0

        G_loss = 1.0 * vgg_loss + 1.0 * recon_loss + 0.0001 * GAN_loss

        G_loss.backward()
        G_optimizer.step()

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = True

        D_optimizer.zero_grad()

        # Train discriminator with real data
        D_real_decision = D(ref)
        # Train discriminator with fake data
        D_fake_decision_1 = D(SR_tar.detach())
        D_fake_decision_2 = D(SR.detach())


        Dis_loss = (BCE_loss(D_real_decision, real_label)
                    + BCE_loss(D_fake_decision_1, fake_label)
                    + BCE_loss(D_fake_decision_2, fake_label)) / 3.0

        # Back propagation
        D_loss = Dis_loss
        D_loss.backward()
        D_optimizer.step()

        # log
        G_epoch_loss += G_loss.data
        D_epoch_loss += D_loss.data
        adv_epoch_loss += (GAN_loss.data)
        recon_epoch_loss += (recon_loss.data)
        vgg_epoch_loss += (vgg_loss.data)

        writer.add_scalars('Train_Loss', {'G_loss': G_loss.data,
                                          'D_loss': D_loss.data,
                                          'VGG_loss': vgg_loss.data,
                                          'Adv_loss': GAN_loss.data,
                                          'Recon_loss': recon_loss.data
                                          }, epoch, iteration)
        print(
            "===> Epoch[{}]({}/{}): G_loss: {:.4f} || D_loss: {:.4f} ||  Adv: {:.4f} || Recon_Loss: {:.4f} || VGG_Loss: {:.4f}".format(
                epoch, iteration,
                len(training_data_loader), G_loss.data, D_loss.data, GAN_loss.data, recon_loss.data, vgg_loss.data))
    print(
        "===> Epoch {} Complete: Avg. G_loss: {:.4f} D_loss: {:.4f} Recon_loss: {:.4f} Adv: {:.4f}".format(
            epoch, G_epoch_loss / len(training_data_loader), D_epoch_loss / len(training_data_loader),
                   recon_epoch_loss / len(training_data_loader),
                   adv_epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch, pretrained_flag=False):
    if pretrained_flag:
        model_out_G = opt.save_folder + opt.model_type + "_pretrain_{}.pth".format(epoch)
        torch.save(G.state_dict(), model_out_G)
        print("Checkpoint saved to {}".format(model_out_G))
    else:
        model_out_G = opt.save_folder + opt.model_type + "_generator_{}.pth".format(epoch)
        model_out_D = opt.save_folder + opt.model_type + "_discriminator_{}.pth".format(epoch)
        torch.save(G.state_dict(), model_out_G)
        torch.save(D.state_dict(), model_out_D)
        print("Checkpoint saved to {} and {}".format(model_out_G, model_out_D))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.upscale_factor, opt.patch_size,
                             opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')

denoiser = VAE_denoise_vali(input_dim=3, dim=32, feat_size=8, z_dim=512, prior='standard')
G = VAE_SR(input_dim=3, dim=64, scale_factor=opt.upscale_factor)
D = discriminator(num_channels=3, base_filter=64, image_size=opt.patch_size * opt.upscale_factor)
feat_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device='cuda')


denoiser = torch.nn.DataParallel(denoiser, device_ids=gpus_list)
G = torch.nn.DataParallel(G, device_ids=gpus_list)
D = torch.nn.DataParallel(D, device_ids=gpus_list)
feat_extractor = torch.nn.DataParallel(feat_extractor, device_ids=gpus_list)


L1_loss = nn.L1Loss()
BCE_loss = nn.BCEWithLogitsLoss()


print('---------- Generator architecture -------------')
print_network(G)
print('---------- Discriminator architecture -------------')
print_network(D)
print('----------------------------------------------')

model_denoiser = os.path.join(opt.save_folder + 'VAE_denoiser.pth')
denoiser.load_state_dict(torch.load(model_denoiser, map_location=lambda storage, loc: storage))
print('Pre-trained Denoiser model is loaded.')

if opt.pretrained:
    model_G = os.path.join(opt.save_folder + opt.pretrained_sr)
    model_D = os.path.join(opt.save_folder + opt.pretrained_D)
    if os.path.exists(model_G):
        G.load_state_dict(torch.load(model_G, map_location=lambda storage, loc: storage))
        print('Pre-trained Generator model is loaded.')
    if os.path.exists(model_D):
        D.load_state_dict(torch.load(model_D, map_location=lambda storage, loc: storage))
        print('Pre-trained Discriminator model is loaded.')

if cuda:
    denoiser = denoiser.cuda(gpus_list[0])
    G = G.cuda(gpus_list[0])
    D = D.cuda(gpus_list[0])
    HR_feat_extractor = HR_feat_extractor.cuda(gpus_list[0])
    feat_extractor = feat_extractor.cuda(gpus_list[0])
    L1_loss = L1_loss.cuda(gpus_list[0])
    BCE_loss = BCE_loss.cuda(gpus_list[0])
    Lap_loss = Lap_loss.cuda(gpus_list[0])


G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)

if opt.pretrain_flag:
    print('Pre-training starts.')
    for epoch in range(1, opt.pretrained_iter + 1):
        train_pretrained(epoch)

        if epoch % 100 == 0:
            checkpoint(epoch, pretrained_flag=True)
    print('Pre-training finished.')

writer = SummaryWriter(opt.log_folder)
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)

    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in G_optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('G: Learning rate decay: lr={}'.format(G_optimizer.param_groups[0]['lr']))
        for param_group in D_optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('D: Learning rate decay: lr={}'.format(D_optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch + 1)
