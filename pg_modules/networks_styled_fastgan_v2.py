# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
from sys import intern
from builtins import super
import torch.nn as nn
import torch
import torch.nn.functional as F
from pg_modules.blocks import (InitLayer, UpBlockBig, UpBlockBigSP, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d)


def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API

class MappingLayers(nn.Module):  # 2 layer-mapping network
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim)

        )
    
    def forward(self, noise):

        return self.mapping(noise)



class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:,:, None, None]
        style_shift = self.style_shift_transform(w)[:,:, None, None]

        transfromed_image = style_scale* normalized_image + style_shift
        return transfromed_image


    # adain = AdaIn (image_channels, w_channels)




class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, w_dim=256, nc=3, img_resolution=256, lite=False):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.w_dim = w_dim

        # channel multiplier
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlockSP = UpBlockSmall if lite else UpBlockBigSP
        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.feat_8   = UpBlockSP(nfc[4], nfc[8])
        self.adain8 = AdaIN(nfc[8], self.w_dim )
        self.feat_16  = UpBlockSP(nfc[8], nfc[16])
        self.adain16  = AdaIN(nfc[16], self.w_dim)
        self.feat_32  = UpBlockSP(nfc[16], nfc[32])
        self.adain32  = AdaIN(nfc[32], self.w_dim)

        self.feat_64  = UpBlockSP(nfc[32], nfc[64])
        self.adain64  = AdaIN(nfc[64], self.w_dim)

        
        self.feat_128 = UpBlockSP(nfc[64], nfc[128])
        self.adain128  = AdaIN(nfc[128], self.w_dim)

        self.feat_256 = UpBlockSP(nfc[128], nfc[256])
        self.adain256  = AdaIN(nfc[256], self.w_dim)

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input, w, c, **kwargs):
        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_8 = self.adain8(feat_8, w)
        feat_16 = self.feat_16(feat_8)
        feat_16 = self.adain16(feat_16, w)
        feat_32 = self.feat_32(feat_16)
        feat_32 = self.adain32(feat_32, w)

        feat_64 =  self.feat_64(feat_32)
        feat_64 =  self.adain64(feat_64, w)
        
        feat_64 = self.se_64(feat_4, feat_64)

        feat_128 = self.feat_128(feat_64)
        feat_128 = self.adain128(feat_128, w)
        feat_128 = self.se_128(feat_8, feat_128 )

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_256 = self.feat_256(feat_last)
            feat_256 = self.adain256(feat_256, w)
            feat_last = self.se_256(feat_16, feat_256)

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)

        return self.to_big(feat_last)


class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, w_dim= 256, nc=3, img_resolution=256, num_classes=1000, lite=False):
        super().__init__()

        self.z_dim = z_dim
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125, 2048:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = img_resolution

        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond

        self.feat_8   = UpBlock(nfc[4], nfc[8], z_dim)
        self.feat_16  = UpBlock(nfc[8], nfc[16], z_dim)
        self.feat_32  = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64  = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.embed = nn.Embedding(num_classes, z_dim)

    def forward(self, input, w, c, update_emas=False):
        c = self.embed(c.argmax(1))

        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4, c)
        feat_16 = self.feat_16(feat_8, c)
        feat_32 = self.feat_32(feat_16, c)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, c))
        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64, c))

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=256,
        map_hidden_dim = 256,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.map_hidden_dim = map_hidden_dim
        

        # Mapping and Synthesis Networks
        self.mapping = DummyMapping()  # to fit the StyleGAN API
        self.our_map = MappingLayers(self.z_dim, self.map_hidden_dim, self.w_dim )
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(ngf=ngf, z_dim=z_dim, w_dim= w_dim, nc=img_channels, img_resolution=img_resolution, **synthesis_kwargs)

    def forward(self, z, c, **kwargs):
        x = self.mapping(z, c)
        w = self.our_map(z)
        img = self.synthesis(x,w, c)
        return img
