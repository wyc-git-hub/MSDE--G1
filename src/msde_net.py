from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops  # 【新增】：引入可变形卷积算子


# ===================================================================== #
# [新增核心模块 1]：全局线性注意力 (Mamba/SSM 的纯原生代理)
# 作用：替代原版 5x5, 7x7 的局部受限卷积，以 O(N) 复杂度捕捉整张眼底图的解剖学先验
# ===================================================================== #
class LinearGlobalAttention(nn.Module):
    """
    O(N) 复杂度的全局线性注意力模块 (Mamba Proxy)。
    通过数学推导将传统的 O(N^2) 注意力转化为 O(N)，无需庞大的显存即可获取全局感受野，
    极度契合眼底图（黄斑、视盘）的空间分布规律。
    """

    def __init__(self, channels):
        super().__init__()
        # 生成 Q, K, V
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        # 将通道切分为 Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 展平空间维度，N = H * W
        q = q.view(b, c, -1)  # [B, C, N]
        k = k.view(b, c, -1)  # [B, C, N]
        v = v.view(b, c, -1)  # [B, C, N]

        # 计算 K 的 Softmax，这是转为线性复杂度的核心
        k = F.softmax(k, dim=-1)

        # 线性注意力矩阵乘法：先算 (K^T * V) -> 得到全局上下文矩阵 [B, C, C]
        context = torch.bmm(k, v.transpose(1, 2))
        # 将全局上下文分发回每一个像素 Q -> [B, C, N]
        out = torch.bmm(context.transpose(1, 2), q)

        # 还原为图像形状
        out = out.view(b, c, h, w)
        out = self.proj(out)
        out = self.norm(out)
        return self.relu(out)


# ===================================================================== #
# [新增核心模块 2]：动态形变卷积分支 (DCN Branch)
# 作用：赋予模型像“流体”一样的软触手，精准贴合出血(HE)和渗出(SE)的极其不规则边缘
# ===================================================================== #
class DCN_Branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        # 解析非对称卷积核的长宽
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # 偏移量生成器 (Offset Generator)：负责观察周围环境，告诉后续卷积核该往哪里扭曲
        # 输出通道数为 2 * kh * kw (每个采样点需要 x 和 y 两个方向的偏移)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kh * kw, kernel_size=3, padding=1, bias=True)
        # 初始化偏移量为0，让它在训练初期先表现得像普通卷积，随后慢慢学习扭曲
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # 核心 DCN 算子
        self.dcn = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 网络自己预测触手的偏移方向
        offsets = self.offset_conv(x)
        # 2. 根据偏移方向进行形变卷积采样
        out = self.dcn(x, offsets)
        return self.relu(self.bn(out))


# ================== 以下为魔改后的 MSDE_Net 主干 ================== #

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None: mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class CMS_Block(nn.Module):
    """
    【升级版 CMS 模块】：引入全局线性感知 (Mamba Proxy)
    """

    def __init__(self, in_channels, out_channels, s=4):
        super(CMS_Block, self).__init__()
        self.s = s

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.width = out_channels // s
        self.branches = nn.ModuleList()

        for i in range(1, s + 1):
            k = 2 * i - 1
            if k == 1:
                # 分支1: 1x1 提取极小病灶特征
                branch = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True)
                )
            elif k == 3:
                # 分支2: 3x3 局部细节特征
                branch = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True)
                )
            else:
                # 【魔改核心】：摒弃 5x5 和 7x7 卷积，全面拥抱 O(N) 全局视野！
                # 并行两个全局注意力头，犹如两只天眼，俯瞰整个黄斑和视盘
                branch = LinearGlobalAttention(self.width)

            self.branches.append(branch)

        self.conv2 = nn.Conv2d(self.width * s, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(x, self.width, 1)

        out = []
        for i in range(self.s):
            out.append(self.branches[i](spx[i]))

        out = torch.cat(out, dim=1)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            CMS_Block(in_channels, out_channels)
        )


class DE_Block(nn.Module):
    """
    【升级版 DE 模块】：赋予 DCN 流体感知能力
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.k_size = kernel_size

        if kernel_size == 1:
            self.eb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        else:
            inner_k = kernel_size - 2
            pad_inner = inner_k // 2
            pad_outer = kernel_size // 2

            # 核心 EB 分支保持普通卷积，维持“靶心”稳定
            self.eb = nn.Conv2d(channels, channels, kernel_size=inner_k, padding=pad_inner, bias=False)

            # 【魔改核心】：将死板的边缘扩张替换为动态寻找病灶边缘的 DCN_Branch
            # EH 水平触手分支
            self.eh = DCN_Branch(channels, channels, kernel_size=(inner_k, kernel_size),
                                 padding=(pad_inner, pad_outer))
            # EV 垂直触手分支
            self.ev = DCN_Branch(channels, channels, kernel_size=(kernel_size, inner_k),
                                 padding=(pad_outer, pad_inner))

    def forward(self, x):
        if self.k_size == 1:
            return self.eb(x)

        x_eb = self.eb(x)
        x_eh = self.eh(x)
        x_ev = self.ev(x)

        return torch.cat([x_eb, x_eh, x_ev], dim=1)


class MSDE_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.de_1 = DE_Block(channels, kernel_size=1)
        self.de_3 = DE_Block(channels, kernel_size=3)
        self.de_5 = DE_Block(channels, kernel_size=5)
        self.de_7 = DE_Block(channels, kernel_size=7)

        concat_channels = channels * 10
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_1 = self.de_1(x)
        out_3 = self.de_3(x)
        out_5 = self.de_5(x)
        out_7 = self.de_7(x)
        out_concat = torch.cat([out_1, out_3, out_5, out_7], dim=1)
        return self.fusion(out_concat)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.msde = MSDE_Module(in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x2 = self.msde(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class MSDENet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,  # 眼底图默认 RGB 3通道
                 num_classes: int = 5,  # 适配你的 IDRiD (背景 + 4类)
                 bilinear: bool = True,
                 base_c: int = 64):
        super(MSDENet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = CMS_Block(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return {"out": logits}