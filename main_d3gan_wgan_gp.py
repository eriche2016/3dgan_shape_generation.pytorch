from __future__ import print_function 
import argparse
import random
import os
import numpy as np 

import torch
import torch.nn as nn 
import torch.nn.parallel # for multi-GPU training 
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch import autograd 
from torch.autograd import Variable 
import models.d3gan as d3gan
import misc.shapenet_vol_dataset as shapenet_vol_dataset  
import misc.utils as utils 

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

parser = argparse.ArgumentParser()

# specify data and datapath 
parser.add_argument('--data_dir', required=True, help='')

# number of workers for loading data
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
# loading data 
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--cube_len', type=int, default=32, help='input cube size, e.g., 32x32x32')

# spicify noise dimension to the Generator 
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')

# spcify optimization stuff 
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--max_epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=1e-3, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')

parser.add_argument('--gpu_id'  , type=str, default='1', help='which gpu to use, used only when ngpu is 1')

parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--lmda', type=int, default=10, help='gradient penalty')

# clamp parameters into a cube 
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument('--gradient_clip', type=float, default=0.01)

parser.add_argument('--experiment', default=None, help='Where to store samples and models')

# resume training from a checkpoint
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--optim_state_from', default='', help="optim state to resume training")


opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'checkpoints_and_samples'

os.system('mkdir {0}'.format(opt.experiment))

# must set this variables before any initialization 
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)

# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 123456

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1: 
        print('so we use 1 gpu to training') 
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

dataset = shapenet_vol_dataset.ShapeNet_Vol_Dataset(opt.data_dir)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

######################################
## from WGAN with gradient penalty, 
## ref: Improved Training of Wasserstein GANs
######################################
def wgan_gp_discriminator_loss(netD, netG, inputv, noisev, lmda): 
    fake = netG(noisev)
    # feed fake data to discriminator 
    errD_fake = netD(fake)
    # calulate fake accuracy 
    # debug_here()
    d_fake_acc = torch.le(errD_fake, 0).float()

    # fake real data to discriminator 
    errD_real = netD(inputv)
    # calculate real accuracy 
    d_real_acc = torch.ge(errD_real, 0).float()

    # total accuracy 
    # debug_here() 
    d_total_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc), 0)).data[0]

    errD = errD_fake.mean() - errD_real.mean() 
    # Wasserstein_D = errD_real - errD_fake

    # draw interpolated values 
    assert(inputv.size(0) == noisev.size(0))
    batch_size = inputv.size(0)
    epsilon = torch.rand(batch_size, 1)
    # interpolates between real and generated data 
    epsilon = epsilon.expand(batch_size, inputv.nelement()//batch_size).contiguous().view(batch_size, inputv.size(1), inputv.size(2), inputv.size(3))
    if inputv.is_cuda:
        epsilon = epsilon.cuda() 
    interpolates = epsilon * inputv.data + ((1 - epsilon) * fake.data)
    interpolates = Variable(interpolates, requires_grad=True)
    # feed into discriminator 
    disc_interpolates = netD(interpolates)
    # get gradients with respect to input 
    if disc_interpolates.is_cuda:
        use_cuda = True 
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    

    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1)**2).mean() * lmda

    # add gradient penalty to the loss 
    errD +=  gradient_penalty

    return errD, errD_real.mean(), errD_fake.mean(), d_total_acc

# custom weights initialization called on netG and netD
# def weights_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Conv') != -1: # classname not contains 'Conv', then return -1, otherwise, return 0 
#        m.weight.data.normal_(0.0, 0.02)
#    elif classname.find('BatchNorm') != -1:
#        m.weight.data.normal_(1.0, 0.02)
#        m.bias.data.fill_(0)

netG = d3gan.D3GAN_G(opt.nz, opt.cube_len)
# doing initialization 
# netG.apply(weights_init)
print(netG)

netD = d3gan.D3GAN_D(opt.cube_len)

print(netD)

# initialize from checkpoints 
if opt.netD != '':
    print('loading checkpoints from {0}'.format(opt.netD))
    netD.load_state_dict(torch.load(opt.netD))
if opt.netG != '': # load checkpoint if needed
    print('loading checkpoints from {0}'.format(opt.netG))
    netG.load_state_dict(torch.load(opt.netG))

fixed_noise = torch.FloatTensor(opt.batch_size, opt.nz).normal_(0, 1)

# shift model and data to GPU 
if opt.cuda:
    netD.cuda()
    netG.cuda()
    fixed_noise = fixed_noise.cuda()


# setup optimizer
# debug_here() 
if opt.adam: # false 
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

# loading optim state 
epoch = 0
if opt.optim_state_from != '':
    print('loading optim_state_from {0}'.format(opt.optim_state_from))
    optim_state = torch.load(opt.optim_state_from)
    epoch = optim_state['epoch']
    # configure optimzer 
    optimizerG.load_state_dict(optim_state['optimizerG_state'])
    optimizerD.load_state_dict(optim_state['optimizerD_state'])

############
# debug_here() 
############
track_d_loss_iter, track_d_loss = [],[]
iter_counter = 0 

gen_iterations = 0
while epoch < opt.max_epochs:
   
    epoch = epoch + 1 
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader): # running one epoch 

        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for p in netG.parameters(): # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
    
        netD.zero_grad() 

        # train the discriminator Diters times
        Diters = 5
        if False: 
            if gen_iterations < 25 or gen_iterations % 50  == 0:
                Diters =500
            else:
                Diters = opt.Diters # 5, i.e., train Determinator 5 
                                    # iterations every 1 iteration training of Generator

        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            i += 1
            
            # clamp parameters to a cube
            # for p in netD.parameters():
            #     p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # load one batch of data 
            data_real = data_iter.next() # i-th batch 
            # train with real
            data_real  = data_real.float() 
            batch_size = data_real.size(0)
        
            mone =  torch.FloatTensor(1).fill_(-1) 
            if opt.cuda:
                mone = Variable(mone.cuda())
                data_real = data_real.cuda()
         
            inputv = Variable(data_real, requires_grad=False)            
            # train with fake
            # Gaussian noise 
            noise = torch.FloatTensor(batch_size, opt.nz).normal_(0, 1)
            if opt.cuda:
                noise = noise.cuda() 
            
            noisev = Variable(noise, requires_grad=False) # totally freeze netG
            # debug_here() 
            errD, errD_real_mean, errD_fake_mean, d_total_acc = wgan_gp_discriminator_loss(netD, netG, inputv, noisev, opt.lmda)
            errD.backward()

            # debug_here()  
            # donot save variables, otherwise, out of memory 
            track_d_loss.append(np.float32(-errD.data[0]))  
            track_d_loss_iter.append(iter_counter)  
            # debug_here() 
            # utils.clip_gradient(optimizerD, opt.gradient_clip)
            # debug_here() 

            if False: 
                optimizerD.step()
            else: 
                # debug_here() 
                if d_total_acc < 0.9:
                    optimizerD.step()

            iter_counter = iter_counter + 1 # recording the steps of training of critic

        ############################
        # (2) Update G network
        ###########################
        for p in netG.parameters(): # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation

        netG.zero_grad()
        
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise = torch.FloatTensor(opt.batch_size, opt.nz).normal_(0, 1)
        mone = torch.FloatTensor(1).fill_(-1)
        if opt.cuda:
            mone = Variable(mone.cuda())
            noisev = Variable(noise.cuda())

        fake = netG(noisev)
        errG = netD(fake) 
        errG = errG.mean()

        errG.backward(mone)   
        G_cost = -errG

        # utils.clip_gradient(optimizerG, opt.gradient_clip)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.max_epochs, i, len(dataloader), gen_iterations,
            errD.data[0], errG.data[0], errD_real_mean.data[0], errD_fake_mean.data[0]))

        if gen_iterations % 50 == 0:
       
            # print('Saving current preal voxel ... ')
            # convert back 
            # real_cpu = real_cpu.mul(0.5).add(0.5)
            ##########
            # debug_here() 
            ##########
            # check https://github.com/pytorch/vision/blob/master/torchvision/utils.py#L81
            # which will convert the tensor type to value of rang (0, 255)
            # utils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(Variable(fixed_noise, volatile=True))
            # take 8 samples
            samples = fake.cpu().data.squeeze().numpy()
            print('Now we begin to generate images ... ')
            utils.SavePloat_Voxels(samples, '{0}/fake_samples/'.format(opt.experiment),  gen_iterations, threshold=0.1)
            # print('save real samples')
            # data_real = data_real.cpu()[:8].squeeze().numpy()
            # utils.SavePloat_Voxels(data_real, '{0}/real_samples/'.format(opt.experiment),  gen_iterations)


    ##############################################################################
    ## saving training info 
    ############################################################################## 
    opt.graph = 5 
    if np.mod(epoch, opt.graph) == 0: 
        print('now we begin to save loss log and its fig ...')
        utils.render_graphs('{0}/loss_log/'.format(opt.experiment),epoch, track_d_loss_iter, track_d_loss) #this will only work after a 50 iterations to allows for proper averating 
        utils.save_values('{0}/loss_log/'.format(opt.experiment),track_d_loss_iter, track_d_loss) # same here but for 300 

    ##############################################################################
    ## save checkpoints every 1 epoch, including netG, netD, and optim_state for optimizer
    ##############################################################################
    # save checkpoint every 1 epoch 
    # do checkpointing
    path_G = '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch%5)
    path_D = '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch%5)
    # save models to checkpoint
    torch.save(netG.state_dict(), path_G)
    torch.save(netD.state_dict(), path_D)
    # save optim_state 
    path_optim_state = '{0}/optim_state_epoch_{1}.pth'.format(opt.experiment, epoch%5)
    optim_state = {} 
    optim_state['epoch'] = epoch 
    # save ids instead of prameters variables 
    # for state, it will save a dictionary of id to variables 
    optim_state['optimizerG_state'] = optimizerG.state_dict() 
    optim_state['optimizerD_state'] = optimizerD.state_dict() 
    torch.save(optim_state, path_optim_state)
