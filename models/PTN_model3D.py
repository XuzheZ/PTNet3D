# Copyright (c) 2021, Xuzhe Zhang, Xinzi He, Yun Wang
# MIT License

from timm.models.registry import register_model
from .transformer_block import Block, get_sinusoid_encoding, Unfold3D, Upfold3D
from torch import nn
import torch
import torch.nn.functional as F
from unfoldNd import UnfoldNd


class Trans_global(nn.Module):
    """
    PTNet class
    """

    def __init__(self, embed_dim=64, depth=9, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        self.sw = UnfoldNd(kernel_size=3, stride=1, padding=1)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=16 ** 3, d_hid=embed_dim), requires_grad=False)


        self.bot_proj = nn.Linear(3 ** 3, embed_dim)
        self.bot_proj2 = nn.Linear(embed_dim, 32)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.bot_proj(x)
        x = x + self.PE
        for blk in self.bottleneck:
            x = blk(x)
        x = self.bot_proj2(x)
        x = x.transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, 16, 16, 16)
        return x


class PTNet_local_trans(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 32, 96, 192],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = Trans_global()
        self.down_blocks = nn.ModuleList(
            [Unfold3D(in_channel=1, out_channel=16, patch=3, stride=1, padding=1),  # 64
             Unfold3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 32
             Unfold3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1),  # 16
             Unfold3D(in_channel=96, out_channel=192, patch=3, stride=2, padding=1)])  # 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [Upfold3D(in_channel=channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(Upfold3D(in_channel=channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [Upfold3D(in_channel=2 * channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(Upfold3D(in_channel=2 * channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(Upfold3D(in_channel=channels[1] + 1,
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1] * patch[-1], embed_dim)
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        Global_feat = self.GlobalGenerator(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=True))
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
                if i == 2:
                    x = torch.cat((x, Global_feat), dim=1)
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-2]):

                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
            for up in self.up_blocks[-2:-1]:
                x = up(x, SC=SC[0], reshape=True,
                       size=[x.shape[2], x.shape[3], x.shape[4]])

            x = self.up_blocks[-1](x, SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        return self.tanh(x)


def PTN_local_trans(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local_trans(img_size=img_size, **kwargs)

    return model


