# Copyright (c) 2021, Xuzhe Zhang, Xinzi He, Yun Wang
# MIT License

from timm.models.registry import register_model
from .transformer_block import Block, get_sinusoid_encoding, Unfold2D, Upfold2D
from torch import nn
import torch
import torch.nn.functional as F


class PTNet(nn.Module):
    """
    PTNet class
    """

    def __init__(self, img_size=[224, 256], trans_type='performer', down_ratio=[1, 1, 2, 4], channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=512, depth=12, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=True):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [Unfold2D(in_channel=channels[i],
                      out_channel=channels[i + 1], patch=patch[i],
                      stride=int(down_ratio[i + 1] / down_ratio[i]),
                      padding=int((patch[i] - 1) / 2)) for i in range(len(down_ratio) - 1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [Upfold2D(in_channel=channels[-(i + 1)],
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(Upfold2D(in_channel=channels[1],
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [Upfold2D(in_channel=
                          int(channels[-(i + 1)] + channels[-(i + 1)] / int(
                              down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]) ** 2),
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(Upfold2D(in_channel=
                                           int(channels[1] + channels[1] / int(down_ratio[2] / down_ratio[1]) ** 2),
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[2] / down_ratio[1]),
                                           padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(Upfold2D(in_channel=int(channels[1] / int(down_ratio[1] / down_ratio[0]) ** 2) + 1,
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                # print(x.shape)
                x = down(x)
                # print(x.shape)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-1]):
                x = up(x, size=[x.shape[2], x.shape[3]], SC=SC[-(i + 1)], reshape=True)
            if not self.individual:
                x = self.up_blocks[-1](x, SC=x0, reshape=True, size=[x.shape[2], x.shape[3]])
            else:
                x = self.up_blocks[-1](x, SC=x0, reshape=False)

        if not self.individual:
            return x
        else:
            x = self.final_proj(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, self.size[0], self.size[1])
            return self.tanh(x)


class PTNet_local(nn.Module):
    def __init__(self, img_size=[224, 256], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 32, 64, 128, 256],
                 patch=[7, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = PTNet(img_size=[int(img_size[0] / 2), int(img_size[1] / 2)],
                                     down_ratio=[1, 1, 2, 4, 8],
                                     channels=[1, 32, 64, 128, 256],
                                     patch=[7, 3, 3, 3, 3], embed_dim=512, depth=9, individual_use=False,
                                     skip_connection=True)
        self.down_blocks = nn.ModuleList(
            [Unfold2D(in_channel=channels[i],
                      out_channel=channels[i + 1], patch=patch[i],
                      stride=int(down_ratio[i + 1] / down_ratio[i]),
                      padding=int((patch[i] - 1) / 2)) for i in range(len(down_ratio) - 1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [Upfold2D(in_channel=channels[-(i + 1)],
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(Upfold2D(in_channel=channels[1],
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [Upfold2D(in_channel=int(
                    channels[-(i + 1)] / int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]) ** 2 + channels[
                        -(i + 1)]),
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(
                Upfold2D(in_channel=int(channels[1] + channels[1] / int(down_ratio[2] / down_ratio[1]) ** 2 + 8),
                         out_channel=channels[1], patch=patch[0],
                         up_scale=int(down_ratio[2] / down_ratio[1]),
                         padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(Upfold2D(in_channel=int(channels[1] / int(down_ratio[1] / down_ratio[0]) ** 2 + 1),
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        Global_res = self.GlobalGenerator(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True))
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-2]):
                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3]])
            for up in self.up_blocks[-2:-1]:
                x = up(torch.cat((x, Global_res), dim=1), SC=SC[0], reshape=True, size=[x.shape[2], x.shape[3]])

            x = self.up_blocks[-1](x, SC=x0, reshape=False)

        x = self.final_proj(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, self.size[0], self.size[1])
        return self.tanh(x)


@register_model
def PTN_local(img_size=[224, 256], **kwargs):
    model = PTNet_local(img_size=img_size, depth=9, **kwargs)

    return model
