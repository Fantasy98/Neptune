import torch 
from torch import nn 
from utils.CBAM import CBAM
from utils.toolbox import periodic_padding
class FCN_Pad_Xaiver_CBAM(nn.Module):
    def __init__(self, height,width,channels,knsize,padding) -> None:
        super(FCN_Pad_Xaiver_CBAM,self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.knsize = knsize
        self.padding = padding
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,out_channels=64,kernel_size=5)
        # self.cb1 = CBAM(64,1,5)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.knsize)
        # self.cb2 = CBAM(128,1,knsize)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=self.knsize)
        self.cb3 = CBAM(256,1,knsize)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=self.knsize)
        self.cb4 = CBAM(256,1,knsize)
        # self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=self.knsize)
        # self.cb5 = CBAM(128,1,knsize)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128,kernel_size=self.knsize)
        self.tcb1 = CBAM(128,1,knsize)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=256+128,out_channels=256,kernel_size=self.knsize)
        self.tcb2 = CBAM(256,1,knsize)
        self.Tconv3 = nn.ConvTranspose2d(in_channels=128+256,out_channels=256,kernel_size=self.knsize)
        # self.tcb3 = CBAM(256,1,knsize)
        self.Tconv4 = nn.ConvTranspose2d(in_channels=64+256,out_channels=64,kernel_size=5)
        # self.tcb4 = CBAM(64,1,5)
        self.out = nn.ConvTranspose2d(in_channels=64+channels,out_channels=1,kernel_size=1)

        self.initial_norm = nn.BatchNorm2d(self.channels,eps=1e-3,momentum=0.99)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.bn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn4 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.bn5 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)
        
        self.tbn1 = nn.BatchNorm2d(128,eps=1e-3,momentum=0.99)  
        self.tbn2 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn3 = nn.BatchNorm2d(256,eps=1e-3,momentum=0.99)  
        self.tbn4 = nn.BatchNorm2d(64,eps=1e-3,momentum=0.99)  

        self.elu = nn.ELU()

   
    def forward(self, inputs):
        padded = periodic_padding(inputs,self.padding)
        batch1 = self.initial_norm(padded)    

        cnn1 = (self.conv1(batch1))
        # cnn1 = self.cb1(cnn1) 
        batch2 =self.elu(self.bn1(cnn1))
        # batch2 = self.cb1(batch2)


        cnn2 = (self.conv2(batch2))
        # cnn2 = self.cb2(cnn2)
        batch3= self.elu(self.bn2(cnn2))
        # batch3 = self.cb2(batch3)

        cnn3 = (self.conv3(batch3))
        cnn3 = self.cb3(cnn3)
        batch4 = self.elu(self.bn3(cnn3))
        # batch4 = self.cb3(batch4)
        
        cnn4 = (self.conv4(batch4))
        cnn4 = self.cb4(cnn4)
        batch5 = self.elu(self.bn4(cnn4))
        # x = self.conv5(batch5)
        # x = self.cb5(x)
        # x = F.elu(self.bn5(x))
        # # batch5 = self.cb4(batch5)
        # x = self.out(x)

        tconv1 = (self.Tconv1(torch.concat([cnn4,batch5],dim=1)))
        # batch6 = self.tcb1(tconv1)
        batch6 = self.elu(self.tbn1(tconv1))
        batch6 = self.tcb1(batch6)

        tconv2 = (self.Tconv2(torch.concat([cnn3,batch6],dim=1)))
        # batch7 = self.tcb2(tconv2)
        batch7 =self.elu(self.tbn2(tconv2))
        batch7 = self.tcb2(batch7)
        
        tconv3 = (self.Tconv3(torch.concat([cnn2,batch7],dim =1 )))
        batch8 = self.elu(self.tbn3(tconv3))
        # batch8 = self.tcb3(batch8)
        
        tconv4 = (self.Tconv4(torch.concat([cnn1,batch8],dim=1)))
        batch9 = self.elu(self.tbn4(tconv4))
        # batch9 = self.tcb4(batch9)
        x = self.out(torch.concat([padded,batch9],dim=1))

        #Corp the padding
        out = x[:,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding]
        
        return out