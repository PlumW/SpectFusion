
from matplotlib import pyplot as plt

from DO_conv import DOConv2d
from doconv_pytorch import DOConv, DOConv_eval
from utils import *
import torch.nn.functional as F
import numpy
from cross_fft import fftatt
from MyModel.REG_new import ConvOffset2D
# x 的shape 应该是 B, 1, H, W
# 根据func_type选择laplacian还是sobel
def laplacian(x, func_type):
    # 转成numpy格式
    x_numpy = x.cpu().detach().numpy()
    # 一张张处理
    result = []
    for i in range(len(x_numpy)):
        img = x_numpy[i][0] # 因为是灰度图，直接取下标0，img就是（H，W）的格式了，就可以放进函数里面了
        if func_type == 'laplacian':
            x_h = cv2.Laplacian(img, -1)
        elif func_type == 'sobel':
            x_h = cv2.Sobel(img, -1, 1, 1)
        else:
            x_h = x
        # 调整shape
        x_h = x_h.reshape(1, 1, x_h.shape[0], x_h.shape[1])
        # 处理完放到数组中，现在这个数组中应该全是(1, 1, H, W)的图片了
        result.append(x_h)
    # 在第一维上面cat一下，变回 B, 1, H, W
    result = numpy.concatenate(result, axis=0)
    # 由numpy再转成tensor
    result = torch.from_numpy(result).cuda()
    return result

class GradRetent(nn.Module):
    def __init__(self):
        super(GradRetent, self).__init__()
        self.fe1 = torch.nn.Conv2d(1, 32, 1, 1, 0)
        self.fe2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.fe3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fe4 = torch.nn.Conv2d(32, 32, 1, 1, 0)

        self.lrelu1 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu2 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu3 = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        l = laplacian(x, 'laplacian')
        l = self.lrelu1(self.fe1(l))

        s1 = laplacian(x, 'sobel')
        s2 = self.lrelu2(self.fe2(s1 + x))
        s3 = self.lrelu3(self.fe3(s2))
        s = self.fe3(s3)
        grad = torch.cat([l, s], dim=1)
        return grad

class HOLOCOModel(nn.Module):
    def __init__(self, batch_size=2):
        super(HOLOCOModel, self).__init__()
        self.batch_size = batch_size
        self.reg = ConvOffset2D()
        self.feat = FeatureEx()
        self.fft1 = fftatt(64)
        self.ftt2 = fftatt(64)
        self.recon = Recontruct()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self, MRI, PET):

        P = self.reg(MRI,PET)
        a,b = self.feat(MRI,P)
        F1 = self.fft1(a,b)
        F2 = self.ftt2(b,a)
        F = torch.cat([F1, F2], dim=1)
        Fusion = self.recon(F)
        return Fusion

class FeatureEx(torch.nn.Module):
    def __init__(self):
        super(FeatureEx, self).__init__()

        self.Do_conv11 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Do_conv12 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.SFB1 = EBlock(64)
        self.SFB2 = EBlock(64)


    def CMDAF(self, vi_feature, ir_feature):
        gap1 = nn.AdaptiveAvgPool2d(1)
        gap2 = nn.AdaptiveAvgPool2d(1)
        batch_size, channels, _, _ = vi_feature.size()

        sub_vi_ir = vi_feature - ir_feature
        vi_ir_div = sub_vi_ir * F.sigmoid(gap1(sub_vi_ir))

        sub_ir_vi = ir_feature - vi_feature
        ir_vi_div = sub_ir_vi * F.sigmoid(gap2(sub_ir_vi))


        # 特征加上各自的带有简易通道注意力机制的互补特征
        vi_feature = vi_feature + ir_vi_div
        ir_feature = ir_feature + vi_ir_div
        return vi_feature, ir_feature

    def Fusion(self, vi_out, ir_out):
        return torch.cat([vi_out, ir_out], dim=1)

    def forward(self, MRI, PET):
        # feature extraction
        att11 = self.Do_conv11(MRI)
        att12 = self.Do_conv12(PET)

        a = self.SFB1(att11)
        b = self.SFB2(att12)


        F1, F2 = self.CMDAF(a, b)
        return F1, F2

class Recontruct(nn.Module):
    def __init__(self):
        super(Recontruct, self).__init__()
        self.conv1 = reflect_conv(in_channels=128, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=32, kernel_size=1, out_channels=1, stride=1, pad=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = nn.Tanh()(self.conv4(x))
        return x
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )
    def forward(self, x):
        out = self.conv(x)
        return out
class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.ConvReLU_1 = BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.ConvReLU_2 = BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=3, stride=1, relu=True)
        self.ConvReLU_3 = BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.Conv3_1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        self.Conv3_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        self.Conv3_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,padding=1)
        self.Conv1_1 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1)
        self.Conv1_2 = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1)
        self.grad = GradRetent()

        self.norm = norm

    def forward(self, x):#x :1*64*64*64
        x_up = self.ConvReLU_1(x) #1*62*62*62
        x_upconv3_1 = self.Conv3_1(x_up)
        #开始正向傅里叶
        _, _, H, W = x_upconv3_1.shape
        dim = 1
        x_fft = torch.fft.rfft2(x_upconv3_1, norm=self.norm)
        x_fft_imag = x_fft.imag
        x_fft_real = x_fft.real
        x_up_f = torch.cat([x_fft_real, x_fft_imag], dim=dim)
        # 正向傅里叶结束
        x_fft = self.ConvReLU_2(x_up_f)
        #开始反向傅里叶
        x_fft_real, x_fft_imag = torch.chunk(x_fft, 2, dim=dim)
        x_fft = torch.complex(x_fft_real, x_fft_imag)
        x_fft = torch.fft.irfft2(x_fft, s=(H, W), norm=self.norm)
        x_fft = x_up+x_fft
        x_fft = self.Conv1_1(x_fft)
        #频域操作结束

        x_down = self.grad(x)
        x = torch.cat([x_fft, x_down], dim=dim)
        x = self.Conv1_2(x)
        return x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=1, ResBlock=ResBlock_do_fft_bench):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
