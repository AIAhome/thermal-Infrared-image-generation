import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):

    def __init__(self, in_channe, out_channel, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(
            in_channe, out_channel, 4, 2,
            1)]  # out = (in+2*padding-kernel-size)/stride+1, i.e. out = in/2
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_channe, out_channel, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channe, out_channel, 4, 2, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]  # out = (in-1)*stride-2*padding+kernel_size i.e. out = 2*in 上采样二倍
        if dropout:  # 只有生成器有dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # 特征融合, feature map在channels层面堆叠

        return x


##############################
#        Gannerator
##############################


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, channel_num=64):
        super(GeneratorUNet, self).__init__()
        # channels, _, _ = input_shape
        self.channel_num = channel_num  # G 基础通道数
        self.down1 = UNetDown(in_channels, self.channel_num,
                              normalize=False)  # channels=3
        self.down2 = UNetDown(self.channel_num, self.channel_num * 2)
        self.down3 = UNetDown(self.channel_num * 2,
                              self.channel_num * 4,
                              dropout=0.5)
        self.down4 = UNetDown(self.channel_num * 4,
                              self.channel_num * 8,
                              dropout=0.5)
        self.down5 = UNetDown(self.channel_num * 8,
                              self.channel_num * 8,
                              dropout=0.5)
        self.down6 = UNetDown(self.channel_num * 8,
                              self.channel_num * 8,
                              dropout=0.5)
        self.down7 = UNetDown(self.channel_num * 8,
                              self.channel_num * 8,
                              dropout=0.5)
        self.down8 = UNetDown(self.channel_num * 8,
                              self.channel_num * 8,
                              dropout=0.5,
                              normalize=False)

        self.up1 = UNetUp(self.channel_num * 8,
                          self.channel_num * 8,
                          dropout=0.5)
        self.up2 = UNetUp(self.channel_num * 8 * 2,
                          self.channel_num * 8,
                          dropout=0.5)
        self.up3 = UNetUp(self.channel_num * 8 * 2,
                          self.channel_num * 8,
                          dropout=0.5)
        self.up4 = UNetUp(self.channel_num * 8 * 2,
                          self.channel_num * 8,
                          dropout=0.5)
        self.up5 = UNetUp(self.channel_num * 8 * 2,
                          self.channel_num * 4,
                          dropout=0.5)
        self.up6 = UNetUp(self.channel_num * 4 * 2, self.channel_num * 2)
        self.up7 = UNetUp(self.channel_num * 2 * 2, self.channel_num)  # 1/2

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(self.channel_num * 2, out_channels, 4, padding=1),
            nn.Tanh()  # 因为有zeropad, 所以Conv2d之后图片的HW与zeropad之前一致
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        # return feature map
        # input (256, 256, in_channel)
        d1 = self.down1(x)  # (128,128, channel_num)
        d2 = self.down2(d1)  # (64, 64, channel_num)
        d3 = self.down3(d2)  # (32, 32, channel_num*2)
        d4 = self.down4(d3)  # (16, 16, channel_num*4)
        d5 = self.down5(d4)  # (8, 8, channel_num*8)
        d6 = self.down6(d5)  # (4, 4, channel_num*8)
        d7 = self.down7(d6)  # (2, 2, channel_num*8)
        d8 = self.down8(d7)  # (1, 1, channel_num*8)

        # 特征融合, feature map在channels层面堆叠
        u1 = self.up1(d8, d7)  # (2, 2, channel_num*8*2)
        u2 = self.up2(u1, d6)  # (4, 4, channel_num*8*2)
        u3 = self.up3(u2, d5)  # (8, 8, channel_num*8*2)
        u4 = self.up4(u3, d4)  # (16, 16, channel_num*8*2)
        u5 = self.up5(u4, d3)  # (32, 32, channel_num*4*2)
        u6 = self.up6(u5, d2)  # (64, 64, channel_num*2*2)
        u7 = self.up7(u6, d1)  # (128, 128, channel_num*2)

        return self.final(u7)  # (256, 256, out_channel)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):

    def __init__(self, input_shape, channel_num=64):
        super(Discriminator, self).__init__()

        self.channel_num = channel_num  # D基础通道数
        channels, height, width = input_shape

        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2**3, width // 2**3
                             )  # out = in/2 three times

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2,
                          padding=1)  # out = in/2
            ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # input (256, 256, channels)
        self.model = nn.Sequential(
            *discriminator_block(
                channels, self.channel_num,
                normalization=False),  # (128, 128, channel_num)
            *discriminator_block(self.channel_num, self.channel_num *
                                 2),  # (64, 64, channel_num*2)
            *discriminator_block(self.channel_num * 2, self.channel_num *
                                 4),  # (32, 32, channel_num*4)
            # Pads the input tensor boundaries with zero. left, right, top, bottom
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(self.channel_num * 4, 1, kernel_size=4,
                      padding=1))  # (32, 32, 1)
        # 输出的是32*32的矩阵, 每个x代表判别器对输入图像一部分(感受野)的判别, 这就是PatchGAN的思想

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
