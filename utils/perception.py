import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def check_image_size(x, padder_size=256):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x


# reflectance-guided loss
def calculate_reflect_map(input, Retinex):
    L, reflect_map = Retinex(input)
    return reflect_map

def calculate_reflect_map_fix(input, Retinex):
    input = torch.pow(input, 0.25)
    data_low = input.squeeze(0) / 20

    data_max_r = data_low[0].max()
    data_max_g = data_low[1].max()
    data_max_b = data_low[2].max()
    
    color_max = torch.zeros((data_low.shape[0], data_low.shape[1], data_low.shape[2])).cuda()
    color_max[0, :, :] = data_max_r * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[1, :, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[2, :, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    
    data_reflect = data_low / (color_max + 1e-6)
    return data_reflect.unsqueeze(0)


# curve exposure loss
def calculate_curve_map(input, Curve):
    _, curve_map, _ = Curve(input)
    return curve_map


# phase gradient loss
class phaseLoss(nn.Module):
    def __init__(self):
        super(phaseLoss, self).__init__()

    def forward(self, input, target):
        H,W = input.shape[-2:]
        x_fft = torch.fft.rfft2(input+1e-8, norm='backward')
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        real_uni = 1 * torch.cos(x_pha)+1e-8
        imag_uni = 1 * torch.sin(x_pha)+1e-8
        x_uni = torch.complex(real_uni, imag_uni)+1e-8
        x_uni = torch.abs(torch.fft.irfft2(x_uni, s=(H, W), norm='backward'))
        x_g = torch.gradient(x_uni,axis=(2,3),edge_order=2)
        x_g_x  = x_g[0];x_g_y = x_g[1]

        y_fft = torch.fft.rfft2(target+1e-8, norm='backward')
        y_amp = torch.abs(y_fft)
        y_pha = torch.angle(y_fft)
        real_uni = 1 * torch.cos(y_pha)+1e-8
        imag_uni = 1 * torch.sin(y_pha)+1e-8
        y_uni = torch.complex(real_uni, imag_uni)+1e-8
        y_uni = torch.abs(torch.fft.irfft2(y_uni, s=(H, W), norm='backward'))
        y_g = torch.gradient(y_uni,axis=(2,3),edge_order=2)
        y_g_x  = y_g[0];y_g_y =y_g[1]

        D_left = torch.pow(x_g_x - y_g_x,2)
        D_right = torch.pow(x_g_y - y_g_y,2)

        E = (D_left + D_right)

        return E.mean()

# color consistency loss
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        k = torch.mean(k)
        
        return k

