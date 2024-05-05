import sys
import numpy as np
import torch

from network.net_util import *
from network.net_vgg import Vgg19
from torch.nn.init import kaiming_normal_, constant_


class Original_YTMT(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=None, reduction=None):
        super(Original_YTMT, self).__init__()
        conv = nn.Conv2d
        self.relu = nn.ReLU()

        self.conv_r = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation,
                                norm=norm, act=None)
        self.conv_t = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation,
                                norm=norm, act=None)

        self.conv_fus_r = ConvLayer(conv, channels + channels, channels, kernel_size=1, stride=1, dilation=dilation,
                                    norm=norm, act=self.relu)
        self.conv_fus_t = ConvLayer(conv, channels + channels, channels, kernel_size=1, stride=1, dilation=dilation,
                                    norm=norm, act=self.relu)

        self.se_r = SELayer(channels, reduction)
        self.se_t = SELayer(channels, reduction)

    def forward(self, list_rt):
        in_r, in_t = list_rt

        r, t = self.conv_r(in_r), self.conv_t(in_t)

        r_p, r_n = self.relu(r), r - self.relu(r)
        t_p, t_n = self.relu(t), t - self.relu(t)

        out_r = torch.cat([r_p, t_n], dim=1)
        out_t = torch.cat([t_p, r_n], dim=1)

        out_r = self.se_r(self.conv_fus_r(out_r))
        out_t = self.se_t(self.conv_fus_t(out_t))

        return out_r, out_t


class Corres_Layer(nn.Module):
    def __init__(self, inter_channels, temperature):
        super(Corres_Layer, self).__init__()
        self.inter_channels = inter_channels
        self.temperature = temperature
        self.theta = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1,
                               stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, m, rs):
        B, C, H, W = m.size()
        theta = self.theta(rs).view(B, self.inter_channels, -1)  # (B, C, HW)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)

        phi = self.phi(m).view(B, self.inter_channels, -1)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)

        theta_permute = theta.permute(0, 2, 1)  # # (B, HW, C)
        phi_permute = phi.permute(0, 2, 1)  # # (B, HW, C)

        # f = torch.matmul(theta_permute, phi)  # (B, HW, HW)
        f = torch.matmul(phi_permute, theta)  # (B, HW, HW)

        confidence_map, _ = torch.max(f, -1, keepdim=True)  # (B, HW, 1)
        confidence_map = confidence_map.view(B, 1, H, W)

        # f can be negative
        f = f / self.temperature
        f_div_C = F.softmax(f, dim=-1)  # (B, HW, HW)

        rs = rs.contiguous().view(B, C, -1)
        rs = rs.permute(0, 2, 1)  # (B, HW, C)

        y = torch.matmul(f_div_C, rs)  # (B, HW, C)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, C, H, W)
        y = torch.mul(y, confidence_map)

        location_map, _ = torch.max(f_div_C, -1, keepdim=True)  # (B, HW, 1)
        location_map = location_map.view(B, 1, H, W)

        return y, location_map


class Net_PAR2Net(nn.Module):
    expansion = 1

    def __init__(self, n_feats=256, n_resblocks=13, patch_num=25, temperature=0.01, reduction=0):
        super(Net_PAR2Net, self).__init__()
        print("===== Use Net_PAR2Net =====")
        self.name = 'Net_PAR2Net'
        self.vgg = Vgg19(requires_grad=False)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        self.patch_num = patch_num
        self.patch_num_per_row = int(np.sqrt(patch_num))
        self.temperature = temperature

        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d

        # Initial other params
        norm = None
        res_scale = 0.1
        if reduction == 0:
            reduction = None

        act = nn.ReLU(inplace=True)

        self.conv_hyper_m = ConvLayer(conv, 1472, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        self.conv_dsample_m = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        self.conv_hyper_rs = ConvLayer(conv, 1472, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        self.conv_dsample_rs = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        n_ytmtblocks = n_resblocks // 2
        res_scale_ytmt = 0.1

        self.res_ytmt = nn.Sequential(*[Original_YTMT(
            n_feats, dilation=1, norm=norm, reduction=reduction) for _ in range(n_ytmtblocks)])

        self.corres_layer = Corres_Layer(inter_channels=n_feats, temperature=self.temperature)
        self.conv_fus_r = ConvLayer(conv, n_feats + n_feats, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        self.res_r = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=1, norm=norm, act=self.relu,
            reduction=reduction, res_scale=res_scale, att_flag='se') for _ in range(n_resblocks - n_ytmtblocks)])

        self.conv_t_weight_map = nn.Sequential(
            ConvLayer(conv, n_feats, n_feats, kernel_size=1, stride=1, norm=norm, act=act),
            ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act),
            ConvLayer(conv, n_feats, 1, kernel_size=1, stride=1, norm=norm, act=self.sigmoid)
        )

        self.conv_fus_t = ConvLayer(conv, n_feats + n_feats, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        self.res_t = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=1, norm=norm, act=self.relu,
            reduction=reduction, res_scale=res_scale, att_flag='se') for _ in range(n_resblocks - n_ytmtblocks)])

        self.decoder_r = nn.Sequential(*[
            ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act),
            ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act),
            PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4),
            ConvLayer(conv, n_feats, 3, kernel_size=1, stride=1, norm=None, act=act)
        ])

        self.decoder_t = nn.Sequential(*[
            ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act),
            ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act),
            PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4),
            ConvLayer(conv, n_feats, 3, kernel_size=1, stride=1, norm=None, act=act)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, m, rs, glass):
        _, C_g, H_g, W_g = glass.shape

        m = m * glass

        m = self.conv_dsample_m(self.conv_hyper_m(self.hyper(m)))
        rs = self.conv_dsample_rs(self.conv_hyper_rs(self.hyper(rs)))

        r, t = self.res_ytmt([m, m.clone()])

        B, C, H, W = r.shape
        conf_map = torch.zeros(size=(B, 1, H, W), requires_grad=True).to(r.device)
        weight_map = torch.zeros(size=(B, 1, H, W), requires_grad=True).to(r.device)
        m_corres = torch.zeros(size=(B, C, H, W), requires_grad=True).to(r.device)

        patch_h, patch_w = H // self.patch_num_per_row, W // self.patch_num_per_row

        for i in range(self.patch_num_per_row - 1):
            for j in range(self.patch_num_per_row - 1):
                patch_m = r[:, :, i * patch_h:(i + 2) * patch_h, j * patch_h:(j + 2) * patch_h]
                patch_rs = rs[:, :, i * patch_h:(i + 2) * patch_h, j * patch_h:(j + 2) * patch_h]

                patch_m, patch_conf_map = self.corres_layer.forward(patch_m, patch_rs)

                weight_map[:, :, i * patch_h:(i + 2) * patch_h, j * patch_h:(j + 2) * patch_h] += 1
                m_corres[:, :, i * patch_h:(i + 2) * patch_h, j * patch_h:(j + 2) * patch_h] += patch_m
                conf_map[:, :, i * patch_h:(i + 2) * patch_h, j * patch_h:(j + 2) * patch_h] += patch_conf_map

        m_corres /= weight_map
        conf_map /= weight_map

        r = torch.cat([r, m_corres], dim=1)
        r = self.conv_fus_r(r)
        r = self.res_r(r)

        residual_t = m - r
        residual_inv_tmap = self.conv_t_weight_map(residual_t)

        res_inv_tmap_dsp = F.interpolate(residual_inv_tmap, size=(H, W), mode='bilinear', align_corners=False)

        t = torch.cat([t, torch.add(residual_t, torch.mul(res_inv_tmap_dsp, residual_t))], dim=1)

        t = self.conv_fus_t(t)
        t = self.res_t(t)

        r = self.decoder_r(r)
        t = self.decoder_t(t)

        r = torch.mul(r, glass)
        t = torch.mul(t, glass)

        r = self.relu(r)
        t = self.relu(t)

        t_weight_map = 1 / (1 + residual_inv_tmap)

        m_rec = torch.mul(t, t_weight_map) + r

        return r, t, m_rec

    def hyper(self, x, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        list_feat = self.vgg(x, indices=indices)
        _, _, H, W = x.size()
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in
                       list_feat]
        hypercolumn = torch.cat(hypercolumn, dim=1)
        return hypercolumn

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
