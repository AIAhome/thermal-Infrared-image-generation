import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.002)
        # torch.nn.init.normal_(m.bias.data, 0.0, 0.002)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.002)
        torch.nn.init.constant_(m.bias.data, 0.0)# ？


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)] # out=(in+2*padding-kernel_size)/stride+1,即这样配置的话出来的时候刚好out=in/2
        if normalize:# 根据原文提供的代码，第一次卷积完了没有BN
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        if dropout:# 只有生成器加dropout
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [#TODO：根据原文提供的代码，为先激活，再逆卷积，最后BN
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
        ]
        if dropout:# 只有生成器有dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1) # U型网络中的跳跃连接，在channel的维度上做拼接

        return x


class Generator(nn.Module):
    def __init__(self,src_channels=3, tgt_channels=3,is_A2B=False):
        super(Generator, self).__init__()
        self.is_A2B = is_A2B
        self.gcn=64# generator channel number
        self.down1 = UNetDown(src_channels, self.gcn, normalize=False)
        self.down2 = UNetDown(self.gcn, self.gcn*2)
        self.down3 = UNetDown(self.gcn*2, self.gcn*4)
        self.down4 = UNetDown(self.gcn*4, self.gcn*8)
        self.down5 = UNetDown(self.gcn*8, self.gcn*8)
        self.down6 = UNetDown(self.gcn*8, self.gcn*8)
        self.down7 = UNetDown(self.gcn*8, self.gcn*8)
        self.down8 = UNetDown(self.gcn*8, self.gcn*8, normalize=False)

        self.up1 = UNetUp(self.gcn*8, self.gcn*8, dropout=0.5)
        self.up2 = UNetUp(self.gcn*16, self.gcn*8, dropout=0.5)
        self.up3 = UNetUp(self.gcn*16, self.gcn*8, dropout=0.5)
        self.up4 = UNetUp(self.gcn*16, self.gcn*8)
        self.up5 = UNetUp(self.gcn*16, self.gcn*4)
        self.up6 = UNetUp(self.gcn*8, self.gcn*2)
        self.up7 = UNetUp(self.gcn*4, self.gcn)

        if is_A2B:
            self.final = nn.Sequential( nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(self.gcn*2, 1, 4, stride=2, padding=1,bias=True),
                                    nn.Tanh(),
                                    # nn.InstanceNorm2d(tgt_channels, affine=True),
                                    )
        else:
            self.final = nn.Sequential( nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(self.gcn*2, tgt_channels, 4, stride=2, padding=1,bias=True),
                                    nn.Tanh(),
                                    # nn.InstanceNorm2d(tgt_channels, affine=True),
                                    )
        

    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        # img is (256 , 256 , self.src_channels)
        d1 = self.down1(x) # (128, 128, self.gcn)
        d2 = self.down2(d1)# (64, 64, self.gcn*2)
        d3 = self.down3(d2)# (32, 32, self.gcn*4)
        d4 = self.down4(d3)# (16, 16, self.gcn*8)
        d5 = self.down5(d4)# (8, 8, self.gcn*8)
        d6 = self.down6(d5)# (4, 4, self.gcn*8)
        d7 = self.down7(d6)# (2 ,2, self.gcn*8)
        d8 = self.down8(d7)# (1, 1, self.gcn*8)

        u1 = self.up1(d8, d7)# (2, 2, self.gcn*8*2)
        u2 = self.up2(u1, d6)# (4, 4, self.gcn*8*2)
        u3 = self.up3(u2, d5)# (8, 8, self.gcn*8*2)
        u4 = self.up4(u3, d4)# (16, 16, self.gcn*8*2)
        u5 = self.up5(u4, d3)# (32, 32, self.gcn*4*2)
        u6 = self.up6(u5, d2)# (64, 64, self.gcn*2*2)
        u7 = self.up7(u6, d1)# (128, 128, self.gcn*2)

        if self.is_A2B:
            _ = self.final(u7)
            return torch.stack([_,_,_],dim=1).squeeze_()
        else:
            return self.final(u7)# (256, 256, self.tgt_channels)
            


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.dcn = 64# discriminator channel number
        # img is (256, 256, src_channels)
        self.model = nn.Sequential(
            *discrimintor_block(in_channels, self.dcn, normalize=False),# (128, 128, self.dcn)
            *discrimintor_block(self.dcn, self.dcn*2),# (64, 64, self.dcn*2)
            *discrimintor_block(self.dcn*2, self.dcn*4),# (32, 32, self.dcn*4)
            *discrimintor_block(self.dcn*4, self.dcn*8),# (16, 16, self.dcn*8)
            *discrimintor_block(self.dcn*8, self.dcn*8),# (8, 8, self.dcn*8)
            nn.ZeroPad2d((1, 0, 1, 0)),#TODO:原文没有做零填充
            nn.Conv2d(self.dcn*8, 1, kernel_size=4)# 注意输出通道数为1，就是patchGAN的实现
        )

    def forward(self, img_1,img_2):
        return F.sigmoid(self.model(img_1)-torch.mean(self.model(img_2)))
    
    def get_raw_output(self,img):
        return self.model(img)


#-----------
# Feature Extractor
#-----------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)