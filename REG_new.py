# 姓名：Plum
# 开发时间：2024/2/26 15:03
import torch.nn.functional as F
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True):
        super(Conv, self).__init__()
        self.use_bias = use_bias
        self.pad = pad

        padding = pad
        if pad_type == 'reflect':
            self.pad_layer = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad_layer = nn.ZeroPad2d(padding)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, bias=use_bias)

    def forward(self, x):
        if self.pad > 0:
            x = self.pad_layer(x)
        x = self.conv(x)
        return x

class ImgResize(nn.Module):
    def __init__(self, scale):
        super(ImgResize, self).__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, oh, ow = x.size()
        new_oh = int(oh * self.scale)
        new_ow = int(ow * self.scale)
        x = F.interpolate(x, size=(new_oh, new_ow), mode='bilinear', align_corners=False)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(ResBlock, self).__init__()
        self.use_bias = use_bias

        self.conv1 = Conv(in_channels, out_channels, kernel=5, stride=1, pad=2, pad_type='reflect')
        self.conv2 = Conv(in_channels, out_channels, kernel=5, stride=1, pad=2, pad_type='reflect')

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)

        x += residual
        return x

class UpSampling(nn.Module):
    def __init__(self, ratio=2):
        super(UpSampling, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        new_height = int(height * self.ratio)
        new_width = int(width * self.ratio)
        x_d = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
        return x_d


def grid_sample_channels(input, grid):
    # 获取输入的形状，分别为：batchsize, 通道数, 高度，宽度
    batchsize, channels, _, _ = input.shape

    # 初始化一个空的结果列表
    result = []

    for c in range(channels):
        # 选取每一通道的数据
        x = input[:, c:c + 1, :, :]
        # 使用grid_sample对每一通道的数据进行重采样
        offset_channel = F.grid_sample(x, grid)
        # 将结果添加到结果列表中
        result.append(offset_channel)

    # 在通道维度上连接结果
    result = torch.cat(result, dim=1)

    return result
class ConvOffset2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, is_training=False):
        super(ConvOffset2D, self).__init__()
        self.is_training = is_training

        self.offset_conv1 =Conv(in_channels*2, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.res_block1 = ResBlock(out_channels, out_channels)

        self.offset_conv2 = Conv(out_channels, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res_block2 = ResBlock(out_channels, out_channels)

        self.offset_conv3 = Conv(out_channels,out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.res_block3 = ResBlock(out_channels, out_channels)

        self.offset_conv5 = Conv(out_channels, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.offset_conv6 = Conv(out_channels*2, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.us1 = UpSampling()
        self.bn5 = nn.BatchNorm2d(out_channels)


        self.offset_conv7 = Conv(out_channels*2, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.us2 = UpSampling()
        self.bn6 = nn.BatchNorm2d(out_channels)

        self.offset_conv8 = Conv(out_channels*2, out_channels, kernel=3, stride=1, pad=1,pad_type='reflect')
        self.us3 = UpSampling()
        self.bn7 = nn.BatchNorm2d(out_channels)

        self.offset_conv10 = Conv(out_channels, 16,kernel=3, stride=1, pad=1,pad_type='reflect')
        self.bn8 = nn.BatchNorm2d(16)

        self.offset_conv11 = Conv(16, 2,kernel=3, stride=1, pad=1,pad_type='reflect')


    def forward(self, x, y):
        # 获取 x 的形状，分别为：batchsize, 通道数, 高度，宽度
        batchsize, channel, oh, ow = x.shape
        # 创建在 -1.0 和 1.0 之间的ow和oh等间距的向量
        dx = torch.linspace(-1.0, 1.0, ow)
        dy = torch.linspace(-1.0, 1.0, oh)
        # 创建均值网格
        xx, yy = torch.meshgrid(dy, dx)
        # 在dim=2维度上扩展xx和yy的维度
        xx = xx.unsqueeze(-1)
        yy = yy.unsqueeze(-1)
        # 在dim=0维度上扩展xx和yy的维度，即增加一个维度
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        # 连接yy和xx
        identity = torch.cat([yy, xx], dim=-1)
        # 使用指定的重复次数重复此张量。 重复的次数是一个 n 维度的张量。 每个张量指定该维度中的重复次数。
        identity = identity.repeat(batchsize, 1, 1, 1).cuda()
        # identity = identity.permute(0, 3, 1, 2)  # 调整通道数到第二维度

        offsets1 = self.offset_conv1(torch.cat([x, y], dim=1))
        offsets1 =self.bn1(offsets1)
        offsets1 = F.relu(offsets1)
        offsets1 = self.res_block1(offsets1)   #8，32，64，64
        offsets1_ds = F.max_pool2d(offsets1, kernel_size=4, stride=2, padding=1)

        offsets2 = self.offset_conv2(offsets1_ds)
        offsets2 = self.bn2(offsets2)
        offsets2 = F.relu(offsets2)
        offsets2 = self.res_block2(offsets2)
        offsets2_ds = F.max_pool2d(offsets2, kernel_size=4, stride=2, padding=1) #8，32，32，32


        offsets3 = self.offset_conv3(offsets2_ds)
        offsets3 = self.bn3(offsets3)
        offsets3 = F.relu(offsets3)
        offsets3 =self.res_block3(offsets3)
        offsets3_ds = F.max_pool2d(offsets3, kernel_size=4, stride=2, padding=1) #8，32，16，16


        offsets5 = self.offset_conv5(offsets3_ds)
        offsets5 = self.bn4(offsets5)
        offsets5 = F.relu(offsets5)

        offsets6 = self.offset_conv6(torch.cat([self.us1(offsets5), offsets3], dim=1))
        offsets6 = self.bn5(offsets6)
        offsets6 = F.relu(offsets6)


        offsets7 = self.offset_conv7(torch.cat([self.us2(offsets6), offsets2], dim=1))
        offsets7 = self.bn6(offsets7)
        offsets7 = F.relu(offsets7)

        offsets8 = self.offset_conv8(torch.cat([self.us3(offsets7), offsets1], dim=1))
        offsets8 = self.bn7(offsets8)
        offsets8 = F.relu(offsets8)

        offsets10 = self.offset_conv10(offsets8)
        offsets10 =self.bn8(offsets10)
        offsets10 = F.relu(offsets10)

        offsets = self.offset_conv11(offsets10)
        offsets = torch.tanh(offsets)
        offsets = offsets.permute(0, 2, 3, 1)  # 调整通道数到第二维度

        x_offset = grid_sample_channels(x, offsets + identity)

        return x_offset

