
import torch
import torch.nn as nn
class FFT2D(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        fft = torch.fft.fft2(x, norm='ortho')
        fft = torch.fft.fftshift(fft)
        return torch.cat((fft.real, fft.imag), dim=1)


class IFFT2D(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        fft = torch.fft.ifftshift(x)
        fft = torch.chunk(fft, chunks=2, dim=1)
        fft = torch.complex(*fft)
        fft = torch.abs(torch.fft.ifft2(fft, norm='ortho'))
        return fft


# class fftatt(nn.Module):
#     def __init__(self, dim):
#         super(fftatt, self).__init__()
#         self.linerQ = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.linerK = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.linerV = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.linerOut = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.linerDft = nn.Conv2d(dim, dim, 1, 1, 0)
#         self.convDft = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)
#         self.fft = FFT2D()
#         self.ifft = IFFT2D()
#         self.softmax = nn.Softmax(dim=1)
#         self.iDft = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
#             IFFT2D()
#         )
#     def forward(self, x):
#         B, C, H, W = x.shape
#         q = self.linerQ(x).view(B, C, -1)
#         dft = self.convDft(self.fft(self.linerDft(x)))
#         k, v = torch.chunk(dft, dim=1, chunks=2)
#         k = self.linerK(k).view(B, C, -1)
#         v = self.linerV(v).view(B, C, -1)
#         att = q @ k.transpose(-1, -2)
#         att = self.softmax(att)
#         out = att @ v
#         out = out.view(B, C, H, W) + self.iDft(dft)
#         return out

class fftatt(nn.Module):
    def __init__(self, dim):
        super(fftatt, self).__init__()
        self.linerQ = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linerK = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linerV = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linerOut = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linerDft = nn.Conv2d(dim, dim, 1, 1, 0)
        self.convDft = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1)
        self.fft = FFT2D()
        self.ifft = IFFT2D()
        self.softmax = nn.Softmax(dim=1)
        self.iDft = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1),
            IFFT2D()
        )
    def forward(self,x,y):
        B, C, H, W = x.shape
        xq = self.linerQ(x).view(B, C, -1)
        dft = self.convDft(self.fft(self.linerDft(y)))

        yk, yv = torch.chunk(dft, dim=1, chunks=2)
        yk = self.linerK(yk).view(B, C, -1)
        yv = self.linerV(yv).view(B, C, -1)
        att = xq @ yk.transpose(-1, -2)

        att = self.softmax(att)
        out = att @ yv
        out = out.view(B, C, H, W) + self.iDft(dft)
        return out
