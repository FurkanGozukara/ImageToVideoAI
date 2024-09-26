import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ).to(dtype=torch.bfloat16),
        nn.PReLU(out_planes).to(dtype=torch.bfloat16),
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ).to(dtype=torch.bfloat16),
        nn.BatchNorm2d(out_planes).to(dtype=torch.bfloat16),
        nn.PReLU(out_planes).to(dtype=torch.bfloat16),
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1).to(dtype=torch.bfloat16),
            nn.PReLU(c // 2).to(dtype=torch.bfloat16),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1).to(dtype=torch.bfloat16),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1).to(dtype=torch.bfloat16),
            nn.PReLU(c // 2).to(dtype=torch.bfloat16),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1).to(dtype=torch.bfloat16),
        )

    def forward(self, x, flow, scale=1):
        x = x.to(dtype=torch.bfloat16)
        flow = flow.to(dtype=torch.bfloat16)
        
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        flow = (
            F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
            )
            * 1.0
            / scale
        )
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = (
            F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * scale
        )
        mask = F.interpolate(
            mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)

    def forward(self, x, scale_list=[4, 2, 1], training=False):
        x = x.to(dtype=torch.bfloat16)
        
        if not training:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        else:
            img0, img1 = x[:, :3], x[:, 3:6]
        
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = torch.zeros(x.size(0), 4, x.size(2), x.size(3), device=x.device, dtype=torch.bfloat16)
        mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=torch.bfloat16)
        
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](
                torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                scale=scale_list[i],
            )
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])

        return flow_list, mask_list[2], merged