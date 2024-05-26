
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class FrequencyCharLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-12, reduction='mean', _reduction_modes=None):
        super(FrequencyCharLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    # def forward(self, pred, target, weight=None, **kwargs):
    #     pred_fft = torch.fft.rfft2(pred, norm='backward')
    #     target_fft = torch.fft.rfft2(target, norm='backward')
    #     diff_real = torch.sqrt((pred_fft.real - target_fft.real)**2 + self.eps)
    #     diff_imag = torch.sqrt((pred_fft.imag - target_fft.imag)**2 + self.eps)
    #     freq_distance = diff_real + diff_imag
    #     return self.loss_weight*torch.mean(freq_distance)
    def forward(self, image_A, image_B,image_fused, weight=None, **kwargs):
        image_A = torch.fft.rfft2(image_A, norm='backward')
        image_B = torch.fft.rfft2(image_B, norm='backward')
        image_fused = torch.fft.rfft2(image_fused, norm='backward')
        diff_real_A = torch.sqrt((image_A.real - image_fused.real)**2 + self.eps)
        diff_real_B = torch.sqrt((image_B.real - image_fused.real) ** 2 + self.eps)
        diff_imag_A = torch.sqrt((image_A.imag - image_fused.imag)**2 + self.eps)
        diff_imag_B = torch.sqrt((image_B.imag - image_fused.imag) ** 2 + self.eps)
        freq_distance = 0.5*(diff_real_A + diff_imag_A)+0.5*(diff_real_B + diff_imag_B)
        return self.loss_weight*torch.mean(freq_distance)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()
        self.L2_loss = torch.nn.MSELoss()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        # Loss_intensity = self.L2_loss(image_fused, intensity_joint)

        return Loss_intensity

class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.Freq = FrequencyCharLoss()

    
    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 =  self.L_Inten(image_A, image_B, image_fused) #20
        loss_gradient = self.L_Grad(image_A, image_B, image_fused) #100
        #loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused)) #50
        loss_fre = 0.03*self.Freq(image_A,image_B,image_fused)
        fusion_loss = loss_l1 + loss_gradient +loss_fre
        return fusion_loss, loss_gradient, loss_l1,loss_fre
