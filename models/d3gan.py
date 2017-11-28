import torch 
import torch.nn as nn 
import torch.nn.parallel 

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

class D3GAN_G(nn.Module):
    def __init__(self, z_dim=200, cube_len=32):
        super(D3GAN_G, self).__init__() 
        self.cube_len = cube_len # 64
        self.z_dim = z_dim

        padd = (0, 0, 0) 
        if self.cube_len == 32: 
            padd = (1, 1, 1) 

        # 200 -> 256 
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(200, self.cube_len*8, kernel_size=4, padding=padd), 
            nn.BatchNorm3d(self.cube_len*8),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.cube_len*4),
            nn.ReLU(True)
            ) 
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.cube_len*2),
            nn.ReLU(True)
            ) 
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.cube_len),
            nn.ReLU(True)
            )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            )
    def forward(self, x):
        """
        x: bz x 200
        out: bz x 64 x 64 x 64 
        """ 
        out = x.view(-1, self.z_dim, 1, 1, 1)
        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out) 
        out = out.view(-1, self.cube_len, self.cube_len, self.cube_len) # bz x 32 x 32 x 32

        return out 


class D3GAN_D(nn.Module):
    def __init__(self, cube_len=32):
        super(D3GAN_D, self).__init__() 
        
        padd = (0, 0, 0)
        self.cube_len = cube_len # 32 
        if self.cube_len == 32:
            padd = (1, 1, 1)

        # original 
        # self.layer1 = nn.Sequential(
        #     nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, padding=(1,1,1)), 
        #     nn.BatchNorm3d(self.cube_len),
        #     nn.LeakyReLU(0.2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, padding=(1,1,1)), 
        #     nn.BatchNorm3d(self.cube_len*2),
        #     nn.LeakyReLU(0.2)
        #     ) 
        # self.layer3 = nn.Sequential(
        #     nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, padding=(1,1,1)), 
        #     nn.BatchNorm3d(self.cube_len*4),
        #     nn.LeakyReLU(0.2)
        #     ) 
        # self.layer4 = nn.Sequential(
        #     nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, padding=(1,1,1)), 
        #     nn.BatchNorm3d(self.cube_len*8),
        #     nn.LeakyReLU(0.2)
        #     )
        
        # without batch normalization
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, padding=(1,1,1)), 
            nn.InstanceNorm3d(self.cube_len),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=2, padding=(1,1,1)), 
            nn.InstanceNorm3d(self.cube_len*2),
            nn.LeakyReLU(0.2, inplace=True)
            ) 
        self.layer3 = nn.Sequential(
            nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, padding=(1,1,1)), 
            nn.InstanceNorm3d(self.cube_len*4),
            nn.LeakyReLU(0.2, inplace=True)
            ) 
        self.layer4 = nn.Sequential(
            nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, padding=(1,1,1)),
            nn.InstanceNorm3d(self.cube_len*8), 
            nn.LeakyReLU(0.2, inplace=True)  
            )
        
        # nn.Tanh not good?
        self.layer5 = nn.Sequential(
            nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, padding=padd),
            )   
    def forward(self, x):
        """
        x: bz x 64 x 64 x 64 
        out: bz x 200 x 1 x 1 x 1 
        """ 
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len) 
        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out) 
        # for wassersetein loss, other wise, remove it 
        # out = out.mean(0)
        # out = out.view(1)
        out = out.view(-1) # size: batch_size

        return out 
