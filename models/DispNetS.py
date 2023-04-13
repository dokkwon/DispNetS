import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_, zeros_

# when image widht and height are multiple of 128, 
# we don't need crop (slicing)
# enable_crop = 1 for training
# enable_crop = 0 for exporting
enable_crop = 0


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU6(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU6(inplace=True)
    )


def predict_disp(in_planes):
    # Use ReLU
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU6(inplace=True)
    )


def upconv(in_planes, out_planes, relu6=True):
    if relu6 == True:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU6(inplace=True)
            #nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )        

# crop_like is not used as long as resolution is 2^n, where n>=8
def crop_like(input, ref):
    if enable_crop == 1:
        assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]
    else:
        return input


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            #nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False)
            #nn.ReLU(inplace=False)
            )
    else:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            #nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
    
            

class MobileV1_Residual(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride = 1, pad=1, dilation=1):
        super(MobileV1_Residual, self).__init__()

        self.stride = stride
        self.downsample = None
        self.conv1 = convbn_dws(inplanes, planes, kernel_size, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, kernel_size, 1, pad, dilation, second_relu=False)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out


class mobileV1(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride = 1, pad=1, dilation=1):
        super(mobileV1, self).__init__()

        #self.stride = stride
        self.conv1 = convbn_dws(inplanes, planes, kernel_size, stride, pad, dilation)

    def forward(self, x):
        out = self.conv1(x)        
        return out

class DispNetS(nn.Module):

    def __init__(self):
        super(DispNetS, self).__init__()

        conv_planes = [64, 128, 256, 512, 1024]
            
        self.conv1   = mobileV1(6,              conv_planes[0], kernel_size=7, pad=3, stride=2)        
        self.conv2   = mobileV1(conv_planes[0], conv_planes[1], kernel_size=5, pad=2, stride=2)
        self.conv3a  = mobileV1(conv_planes[1], conv_planes[2], kernel_size=5, pad=2, stride=2)
        self.conv3b  = mobileV1(conv_planes[2], conv_planes[2])
        self.conv4a  = mobileV1(conv_planes[2], conv_planes[3], stride=2)
        self.conv4b  = mobileV1(conv_planes[3], conv_planes[3])
        self.conv5a  = mobileV1(conv_planes[3], conv_planes[3], stride=2)
        self.conv5b  = mobileV1(conv_planes[3], conv_planes[3])
        self.conv6a  = mobileV1(conv_planes[3], conv_planes[4], stride=2)
        self.conv6b  = mobileV1(conv_planes[4], conv_planes[4])

        upconv_planes = [512, 256, 128, 64, 32, 16]
        self.upconv6 = upconv(conv_planes[4],   upconv_planes[0])
        self.upconv5 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv4 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv3 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv2 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv1 = upconv(upconv_planes[4], upconv_planes[5])

        self.dispupconv = upconv(1, 1, relu6=False)

        self.iconv5 = mobileV1(upconv_planes[0] + conv_planes[3] + 1, upconv_planes[0])
        self.iconv4 = mobileV1(upconv_planes[1] + conv_planes[3] + 1, upconv_planes[1])
        self.iconv3 = mobileV1(upconv_planes[2] + conv_planes[2] + 1, upconv_planes[2])
        self.iconv2 = mobileV1(upconv_planes[3] + conv_planes[1] + 1, upconv_planes[3])
        self.iconv1 = mobileV1(upconv_planes[4] + conv_planes[0] + 1, upconv_planes[4])

        self.predict_disp7 = predict_disp(conv_planes[4])
        self.predict_disp6 = predict_disp(upconv_planes[0])
        self.predict_disp5 = predict_disp(upconv_planes[1])
        self.predict_disp4 = predict_disp(upconv_planes[2])
        self.predict_disp3 = predict_disp(upconv_planes[3])
        self.predict_disp2 = predict_disp(upconv_planes[4])
        self.predict_disp1 = predict_disp(upconv_planes[5])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)

        out_conv1  = self.conv1(x)
        out_conv2  = self.conv2(out_conv1)
        out_conv3a = self.conv3a(out_conv2)
        out_conv3b = self.conv3b(out_conv3a)
        out_conv4a = self.conv4a(out_conv3b)
        out_conv4b = self.conv4b(out_conv4a)
        out_conv5a = self.conv5a(out_conv4b)
        out_conv5b = self.conv5b(out_conv5a)
        out_conv6a = self.conv6a(out_conv5b)
        out_conv6b = self.conv6b(out_conv6a)
        
        disp7       = self.predict_disp7(out_conv6b)                      # ch: 1024 -> 1
        up_disp7    = crop_like(self.dispupconv(disp7), out_conv5b)
        out_upconv6 = crop_like(self.upconv6(out_conv6b), out_conv5b)     # ch: 512 
        concat5     = torch.cat((out_upconv6, out_conv5b, up_disp7), 1)   # ch: 512+512+1 
        out_iconv5  = self.iconv5(concat5)                                # ch: 512

        disp6       = self.predict_disp6(out_iconv5)                      # ch: 512 -> 1
        up_disp6    = crop_like(self.dispupconv(disp6), out_conv4b)
        out_upconv5 = crop_like(self.upconv5(out_iconv5), out_conv4b)     # ch: 256
        concat4     = torch.cat((out_upconv5, out_conv4b, up_disp6), 1)   # ch: 256+512+1 
        out_iconv4  = self.iconv4(concat4)                                # ch: 256

        disp5       = self.predict_disp5(out_iconv4)                      # ch: 256 -> 1
        up_disp5    = crop_like(self.dispupconv(disp5), out_conv3b)
        out_upconv4 = crop_like(self.upconv4(out_iconv4), out_conv3b)     # ch: 128
        concat3     = torch.cat((out_upconv4, out_conv3b, up_disp5), 1)   # ch: 128+256+1 
        out_iconv3  = self.iconv3(concat3)                                # ch: 128

        disp4       = self.predict_disp4(out_iconv3)                      # ch: 128 -> 1
        up_disp4    = crop_like(self.dispupconv(disp4), out_conv2)
        out_upconv3 = crop_like(self.upconv3(out_iconv3), out_conv2)      # ch: 64
        concat2     = torch.cat((out_upconv3, out_conv2, up_disp4), 1)    # ch: 64+128+1 
        out_iconv2  = self.iconv2(concat2)                                # ch: 64

        disp3       = self.predict_disp3(out_iconv2)                      # ch: 64 -> 1
        up_disp3    = crop_like(self.dispupconv(disp3), out_conv1)
        out_upconv2 = crop_like(self.upconv2(out_iconv2), out_conv1)      # ch: 32
        concat1     = torch.cat((out_upconv2, out_conv1, up_disp3), 1)    # ch: 32+64+1 
        out_iconv1  = self.iconv1(concat1)                                # ch: 32

        disp2       = self.predict_disp2(out_iconv1)                      # ch: 32 -> 1
        disp1       = self.upconv1(out_iconv1)
        disp1       = self.predict_disp1(disp1)

        if self.training:
            # squeeze and return                    
            return [disp1, disp2, disp3, disp4]
        else:            
            return [disp1]
