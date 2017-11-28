import torch 
import torch.nn as nn 
from torch.autograd import Variable

import models.d3gan as d3gan 
import misc.shapenet_vol_dataset as shapenet_vol_dataset  

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

# generator 
netG = d3gan.D3GAN_G() 
# discriminator 
netD = d3gan.D3GAN_D() 

input_noise = Variable(torch.rand(2, 200))
debug_here() 
out = netG(input_noise)

pred = netD(out) 

dataset = shapenet_vol_dataset.ShapeNet_Vol_Dataset('./data/train/chair')
data = dataset[1]
