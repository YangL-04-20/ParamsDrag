# Copyright 2024 The ParamsDrag Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# main file for training
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import visdom

import sys
import time
from dataset512 import *
from utils import *
from network import Generator


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="ParamsDrag training model")
    
    parser.add_argument("--======== 0 ========", type=str, default="training file")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=True,
                        help="enable data parallelism")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")

    parser.add_argument("--root", type=str, default="./dataset",
                        help="root of the dataset")
    parser.add_argument("--store", type=str, default="",
                        help="store in this dir")
    parser.add_argument("--resume", type=str, default="none",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--======== 1 ========", type=str, default="model config")

    parser.add_argument("--resolution", type=int, default=512,
                        help="img resolution")
    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--dvp", type=int, default=3,
                        help="dimensions of the view parameters (default: 3)")
    parser.add_argument("--d-latent", type=int, default=512,
                        help="dimensions of the latent vector, such as z and w(default: 512)")
    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")
    
    parser.add_argument("--======== 2 ========", type=str, default="loss list")
    
    parser.add_argument("--l1-loss", action="store_true", default=False,
                        help="enable l1 loss")
    parser.add_argument("--mse-loss", action="store_true", default=False,
                        help="enable mse loss")
    parser.add_argument("--edge-loss", action="store_true", default=False,
                        help="enable edge loss")
    parser.add_argument("--perc-loss", type=str, default="none",
                        help="layer that perceptual loss is computed on (default: relu1_2 / none)")
    parser.add_argument("--perc-style", type=str, default="mse",
                        help="mse-percloss or l1-percloss")
    
    parser.add_argument("--======== 3 ========", type=str, default="parameter list")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 of Adam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--l1-weight", type=float, default=1.0,
                        help="weight of the l1-loss")
    parser.add_argument("--edge-weight", type=float, default=0.01,
                        help="weight of the edge-loss")
    parser.add_argument("--perc-weight", type=float, default=1.0,
                        help="weight of the perc-loss")
    
    parser.add_argument("--======== 4 ========", type=str, default="training config")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="batch size for training (default: 16)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of epochs to train (default: 1000)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="log training status every given number of batches (default: 10)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="save images every given number of epochs (default: 10)")
    parser.add_argument("--check-every", type=int, default=20,
                        help="pickle checkpoint every given number of epochs (default: 20)")
    parser.add_argument("--vision", type=int, default=0000,
                        help="visdom visulization")

    return parser.parse_args()


# the main function
def main(args):

    # log hyperparameters
    for key, value in vars(args).items():
        print(key, ":", value)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print("===================  Use", torch.cuda.device_count(), 'gpus  ====================')

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data loader
    train_dataset = Dataset(root=args.root, train=True,
        transform=transforms.Compose([Resize(args.resolution), Normalize(), ToTensor()]))
    test_dataset = Dataset(root=args.root, train=False,
        transform=transforms.Compose([Resize(args.resolution), Normalize(), ToTensor()]))

    kwargs = {"num_workers": 16, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, **kwargs)

    # mean and std
    mean, std = getStat(train_dataset)
    print("mean: ",mean, "std: ", std)
    norm_mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    norm_std = torch.tensor(std).view(-1, 1, 1).to(device)

    # define model
    G = Generator(args.d_latent, args.d_latent, args.resolution, args.dsp, args.dvp)
    G.apply(weights_init)
    if args.sn:
        G = add_sn(G)
    if args.data_parallel and torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
    G.to(device)
    # define optimizer
    paramap_optimizer = optim.Adam(G.module.paramap.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    mapping_optimizer = optim.Adam(G.module.mapping.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    synthes_optimizer = optim.Adam(G.module.synthesis.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    model = [G]
    opt = [paramap_optimizer, mapping_optimizer, synthes_optimizer]
    
    def run_G(sp, vp):
        z = G.module.paramap(sp, vp)
        ws = G.module.mapping(z)
        img = G.module.synthesis(ws)
        return img, ws, z

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            G.load_state_dict(checkpoint["G_model"])
            print("=> loaded checkpoint")

    # loss
    train_losses, test_losses = [], []
    L1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    if args.perc_loss != "none":
        vgg = VGG19(args.perc_loss).eval()
        if args.data_parallel and torch.cuda.device_count() > 1:
            vgg = nn.DataParallel(vgg)
        vgg.to(device)

    # main loop
    print(model)
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        train_loss = 0.
        if epoch == 10:
            args.batch_size = 8
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if epoch == 30:
            args.batch_size = 4
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if (epoch + 1) % 200 == 0:
            args.lr = args.lr * 0.5
            paramap_optimizer = optim.Adam(G.module.paramap.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            mapping_optimizer = optim.Adam(G.module.mapping.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            synthes_optimizer = optim.Adam(G.module.synthesis.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            opt = [paramap_optimizer, mapping_optimizer, synthes_optimizer]
            print("Learning rate update to", args.lr)

        for i, sample in enumerate(train_loader):
            image = sample["image"].to(device)
            sparams = sample["sparams"].to(device)
            vparams = sample["vparams"].to(device)
            loss = 0.
            set_requires_grad(model, True)
            set_zerograd(opt)

            fake_image, _, _ = run_G(sp=sparams, vp=vparams)

            image = (image - norm_mean) / norm_std
            fake_image = (fake_image - norm_mean) / norm_std
            
            if args.l1_loss:
                l1_loss = L1_criterion(image, fake_image)
                loss += l1_loss * args.l1_weight

            if args.mse_loss:
                mse_loss = mse_criterion(image, fake_image)
                loss += mse_loss

            if args.edge_loss:
                # Sobel operator is enough
                trans = transforms.Grayscale()
                kernel1 = torch.tensor([[1, 1, -1], [1, 0, -1], [1, 0, -1]]).float().to(device)  # vertical
                kernel2 = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).float().to(device)  # parallel
                kernel1 = (kernel1.unsqueeze(0)).unsqueeze(0)
                kernel2 = (kernel2.unsqueeze(0)).unsqueeze(0)
                real_edge = F.conv2d(F.conv2d(trans(image), kernel1), kernel2)
                fake_edge = F.conv2d(F.conv2d(trans(fake_image), kernel1), kernel2)
                edge_loss = L1_criterion(real_edge, fake_edge)
                loss += edge_loss * args.edge_weight
                # canny-algorithm is better but too slow
                # trans = transforms.Grayscale()
                # real_edge = canny_edge_detection(trans(image), 0.001, 100, 145)
                # fake_edge = canny_edge_detection(trans(fake_image), 0.001, 100, 145)
                # edge_loss = L1_criterion(real_edge, fake_edge)
                # loss += edge_loss * args.edge_weight

            if args.perc_loss != "none":
                features = vgg(image)
                fake_features = vgg(fake_image)
                if args.perc_style == "l1":
                    perc_loss = L1_criterion(features, fake_features)
                else:
                    perc_loss = mse_criterion(features, fake_features)
                loss += perc_loss * args.perc_weight

            loss.backward()
            set_requires_grad(model, False)
            remove_inf(model)
            set_step(opt)

            train_loss += loss.item() * len(sparams)

            # log training status
            if i % args.log_every == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tTime: {:.4f}"
                      .format(epoch, i * args.batch_size, len(train_loader.dataset),
                              100. * i / len(train_loader), loss.item() ,time.time() - start))

        train_losses.append(train_loss / len(train_loader.dataset))
        print("====> Epoch: {}\tAverage loss: {:.8f}\tTime: {:.4f}".format(epoch, train_losses[-1], time.time()-start))

        # testing...
        G.eval()
        test_loss = 0.
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                image = sample["image"].to(device)
                sparams = sample["sparams"].to(device)
                vparams = sample["vparams"].to(device)
                fake_image = G(sp=sparams, vp=vparams)
                test_loss += mse_criterion(image, fake_image).item() * len(sparams)

                if (epoch+1) % args.save_every == 0 and i == 0 :
                    n = min(8, len(sparams))
                    comparison = torch.cat([image[:n], fake_image[:n]])
                    save_image(((comparison.cpu() + 1.) * .5), args.store + "/lr:" + str(args.lr) + "_perc:" + args.perc_loss + \
                        "_edge:" +str(args.edge_loss) +"_l1:" + str(args.l1_loss)+ "_" + str(epoch) + ".png", nrow=n)

        test_losses.append(test_loss / len(test_loader.dataset))
        print("====> Epoch: {} Test set loss: {:.8f}".format(epoch, test_losses[-1]))

        # =========================================================================
        if args.vision != 0000:
            if epoch == 0:
                vis = visdom.Visdom(port=args.vision)
                vis.text('<br>'.join([str(key) + ":" + str(value) for key, value in vars(args).items()]))
            vis.line(
                X = np.column_stack(([epoch],
                                    [epoch])),
                Y = np.column_stack(([np.array(train_losses[-1])],
                                    [np.array(test_losses[-1])])),
                win = "Loss Line",
                update = 'append',
                opts={
                    'legend': ["train loss", "test loss"],
                    'showlegend': True,
                },
            )
            num = 4
            comparison = torch.cat([image[0:num], torch.clamp(fake_image[0:num].view(num, 3, args.resolution, args.resolution),-1,1)])
            vis.images(
                ((comparison+1.)*.5)*255,
                win = "real_images",
                nrow = num
            )
        # =========================================================================

        # saving...
        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch))
            save_name = args.store + "/B" + str(args.batch_size)+ "_lr:" + str(args.lr) + "_perc:" + args.perc_loss + "_edge:" +str(args.edge_loss) +"_l1:" + str(args.l1_loss)+ "_" + str(epoch) + ".pth"
            torch.save({"G_model": G.state_dict()}, save_name)


if __name__ == "__main__":
    main(parse_args())
