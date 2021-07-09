""" Full assembly of the parts to form the complete network """



from .msnet_parts import *


class msNet(nn.Module):
    def __init__(self, n_channels=1, n_sides=5):
        super(msNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_sides = n_sides
        

        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = Down(64, 128) #1/2
        self.down3 = Down3(128, 256) #1/4
        self.down4 = Down3(256, 512) #1/8
        self.down5 = Down3(512, 512) #1/16

        
        # Upsample
        # self.up1=OutConv(64)
        # self.up2 = Up(128,stride=2)
        # self.up3 = Up(256,stride=4)
        # self.up4 = Up(512,stride=8)
        # self.up5 = Up(512,stride=16)
        
        # Upsample
        self.up1=OutConv(64)
        self.up2 = Up(128,upscale=1)#stride=2
        self.up3 = Up(256,upscale=2)#stride=4
        self.up4 = Up(512,upscale=3)#stride=8
        self.up5 = Up(512,upscale=4)#stride=16
        
        self.fuse = OutConv(n_sides, uniform=True)
        
        

    def forward(self, data):

        x=data['image']
        crop_size=(x.size(2), x.size(3)) #(h,w)
        
        down1 = self.down1(x) #64
        down2 = self.down2(down1)#1/2 #128
        down3 = self.down3(down2)#1/4 #256
        down4 = self.down4(down3)#1/8 #512
        down5 = self.down5(down4)#1/16 #512
        
        so1 = self.up1(down1) # size: [1,1,H,W]
        so2 = self.up2(down2, crop_size)
        so3 = self.up3(down3, crop_size)
        so4 = self.up4(down4, crop_size)
        so5 = self.up5(down5, crop_size)
        
    
        
        
        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.fuse(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        
        return results
