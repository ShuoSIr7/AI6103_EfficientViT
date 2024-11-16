"""
author: lqs
12 Nov 2024
"""
import torch
from torch import nn

"""
The EfficientViT model structure leverage LeViT instead of ViT.
Using 2D embeddings instead of 1D, all the computations are based on this structure.

Tricks/Details not mentioned in the essay:
√ 1. Weights initialization.
    bn_weight = 0: 4 places
√ 2. SE Block in InvertedResidualBlock.
√ 3. Local Window Attention.
√ 4. Extra activation before Concat&Projection in CGA.
√ 5. Attention Bias.
× 6. Knowledge Distillation from a teacher model.
"""


class Conv_BN(nn.Module):
    """
    Folded BatchNorm to Conv to increase inference speed
    input: B*C*H*W
    """

    # TODO: I don't know how to initialize gamma in BN, so I keep the parameter but don't use it.
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bn_init_weight=1.):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.fused_conv = None
        # initial parameters
        # default Kaiming initialization for Conv layer
        # initialize BN's gamma & beta
        nn.init.constant_(self.bn.weight, bn_init_weight)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if self.fused_conv is not None:
            return self.fused_conv(x)
        return self.bn(self.conv(x))

    # call this before inference
    @torch.no_grad()
    def fuse(self):
        bn = self.bn
        conv = self.conv
        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True
        )
        weight_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_conv.weight = nn.Parameter(conv.weight * weight_bn.view(-1, 1, 1, 1))
        fused_conv.bias = nn.Parameter(bn.bias - bn.running_mean * weight_bn)
        self.fused_conv = fused_conv

    # call this before training
    def reset_fuse(self):
        self.fused_conv = None


class BN_Linear(nn.Module):
    """
    BN and Linear Layer for MLP Head
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.fused_linear = None
        # initialize linear as what ViT does
        nn.init.trunc_normal_(self.linear.weight, std=.02)

    def forward(self, x):
        if self.fused_linear is not None:
            return self.fused_linear(x)
        return self.bn(self.linear(x))

    @torch.no_grad()
    def fuse(self):
        if self.fused_linear is None:
            bn = self.bn
            linear = self.linear
            fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)

            weight_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias_bn = bn.bias - bn.running_mean * weight_bn

            fused_linear.weight = nn.Parameter(linear.weight * weight_bn.view(1, -1))
            fused_linear.bias = nn.Parameter(linear.bias + torch.matmul(linear.weight, bias_bn))
            self.fused_linear = fused_linear

    def reset_fuse(self):
        self.fused_linear = None


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, channel_scale=1/16):
        super().__init__()
        hid_channels = max(1, int(in_channels * channel_scale))
        self.add_module('ap', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('fc1', nn.Conv2d(
            in_channels, hid_channels, 1, 1, 0, 1, bias=True))
        self.add_module('act1', nn.ReLU())
        self.add_module('fc2', nn.Conv2d(
            hid_channels, in_channels, 1, 1, 0, 1, bias=True))
        self.add_module('act2', nn.Sigmoid())

    def forward(self, x):
        scale = self.ap(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return scale * x


class OverlapPatchEmbedding(nn.Module):
    """
    ref: Levit: a vision transformer in convnet’s clothing for faster inference.

    Overlap patch embedding to represent images.
    input: B*C_in*H*W
    output: B*C_out*(H/16)*(W/16)
    """

    def __init__(self, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            Conv_BN(in_channels, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Conv_BN(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Conv_BN(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Conv_BN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Residual(nn.Module):
    """
    Residual connection with dropout.
    """

    def __init__(self, f, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.f = f

    def forward(self, x):
        if self.training and self.dropout > 0:
            mask = torch.rand(x.shape, device=x.device)
            x = x + self.f(x) * mask.ge_(self.dropout).div(1 - self.dropout).detach()
        else:
            x = x + self.f(x)
        return x


class FFN(nn.Module):
    """
    Feed Forward Layer base on a 2D structure, using Pointwise Conv instead of Linear.
    """

    def __init__(self, channels):
        super(FFN, self).__init__()
        self.hidden_dim = int(2 * channels)
        self.conv1 = Conv_BN(channels, self.hidden_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv_BN(self.hidden_dim, channels, 1, 1, 0, bn_init_weight=0.)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class InvertedResidualBlock(nn.Module):
    """
    Depthwise convolution for subsampling
    Residual connection is not available because H*W doesn't match
    B*C_in*H*W -> B*C_out*(H/2)*(W/2)
    """

    def __init__(self, in_channels, out_channels):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(in_channels * 4)
        self.conv1 = Conv_BN(in_channels, hidden_dim, 1, 1, 0)
        self.act1 = nn.ReLU()
        self.conv2 = Conv_BN(hidden_dim, hidden_dim, 3, 2, 1, groups=hidden_dim)
        # using ReLU instead of SE in LeVit
        # self.act2 = nn.ReLU()
        self.se = SqueezeExcite(hidden_dim, 1/4)
        self.conv3 = Conv_BN(hidden_dim, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv3(self.act2(self.conv2(self.act1(self.conv1(x)))))


class TokenInteractionBlock(nn.Module):
    """
    Token interaction through depthwise convolution.
    Default kernel size is 3, but may be different in different places. (After Attention Query or Before FFN)
    """

    def __init__(self, channels, kernel_size=3, bn_init_weight=1.):
        super(TokenInteractionBlock, self).__init__()
        # padding is set to [kernel//2] to make sure input and output share same dimension
        self.dwconv = Conv_BN(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels, bn_init_weight=bn_init_weight)

    def forward(self, x):
        return self.dwconv(x)


class SubSamplingBlock(nn.Module):
    """
    SubSamplingBlock between EfficientVitBlocks
    B*C_in*H*W -> B*C_out*(H/2)*(W/2)
    """

    def __init__(self, in_channels, output_channels, ffn_depth):
        super().__init__()
        self.ffn_depth = ffn_depth
        self.interact1 = [Residual(TokenInteractionBlock(in_channels))] * self.ffn_depth
        self.ffn1 = [Residual(FFN(in_channels))] * self.ffn_depth
        # res connection unavailable due to different input and output channel
        self.dwconv = InvertedResidualBlock(in_channels, output_channels)
        self.interact2 = [Residual(TokenInteractionBlock(output_channels))] * self.ffn_depth
        self.ffn2 = [Residual(FFN(output_channels))] * self.ffn_depth

    def forward(self, x):
        for i in range(self.ffn_depth):
            x = self.interact1[i](x)
            x = self.ffn1[i](x)
        x = self.dwconv(x)
        for i in range(self.ffn_depth):
            x = self.interact2[i](x)
            x = self.ffn2[i](x)
        return x

"""
LocalWindowAttention
author: cxk
"""

class LocalWindowAttention(torch.nn.Module):
    def __init__(self, channels, qk_dim, v_dim, num_heads=8,
                 resolution=14,
                 window_resolution=7,
                 q_kernel_size=[5, 5, 5, 5]):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(channels, qk_dim, v_dim=v_dim,
                                           num_heads=num_heads, resolution=window_resolution,
                                           q_kernel_size=q_kernel_size)

    def forward(self, x):
        H = W = self.resolution
        h = w = self.window_resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= h and W <= w:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)  # B,H,W,C
            pad_b = (h - H % h) % h
            pad_r = (w - W % w) % w
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            H, W = H + pad_b, W + pad_r
            nH,nW = H // h, W // w
            # window partition, BHWC -> B(nH*h)(nW*w)C -> B,nH,nW,h,w,C -> (BnHnW)hwC -> (BnHnW)Chw
            x = (
                x.view(B, nH, h, nW, w, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, h, w, C)
                .permute(0, 3, 1, 2)
            )
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = (
                x.permute(0, 2, 3, 1)
                .view(B, nH, nW, w, w, C)
                .transpose(2, 3)
                .reshape(B, H, W, C)
            )
            if padding:
                x = x[:, :H-pad_b, :W-pad_r].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class CascadedGroupAttention(nn.Module):
    def __init__(self, channels, qk_dim, v_dim, num_heads, q_kernel_size, resolution):
        super(CascadedGroupAttention, self).__init__()
        self.channels = channels
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        # pre_calculate
        self.sqrtd = self.qk_dim ** -0.5
        assert self.channels % self.num_heads == 0, "channels should be divisible by group"
        self.att_dim = self.channels // self.num_heads
        self.q = []
        self.dwconv = []
        self.k = []
        self.v = []
        self.p = Conv_BN(self.num_heads * self.v_dim, channels, bn_init_weight=0.)
        self.act = nn.ReLU()

        for _ in range(num_heads):
            self.q.append(Conv_BN(self.att_dim, self.qk_dim))
            self.k.append(Conv_BN(self.att_dim, self.qk_dim))
            self.v.append(Conv_BN(self.att_dim, self.v_dim))
            self.dwconv.append(TokenInteractionBlock(self.qk_dim, q_kernel_size))

        # calc relative position
        # gen coordinates
        num_points = resolution * resolution
        h_list = torch.arange(resolution)
        w_list = torch.arange(resolution)
        coords = torch.meshgrid([h_list, w_list])
        coords = torch.stack(coords)
        coords = torch.flatten(coords, 1)
        coords1 = coords[:, :, None]
        coords2 = coords[:, None, :]
        relative_coords = abs(coords1 - coords2)
        # relative positions denote by index
        relative_positions = []
        # relative positions map to index
        relative_positions_index = {}
        cnt = 0
        for i in torch.arange(resolution):
            for j in torch.arange(resolution):
                cur_rc = (relative_coords[0][i][j], relative_coords[1][i][j])
                if cur_rc not in relative_positions_index:
                    relative_positions_index[cur_rc] = cnt
                    cnt += 1
                relative_positions.append(cnt)
        # trainable parameter, maps index to attention bias
        self.attention_bias = nn.Parameter(torch.zeros(num_heads, len(relative_positions_index)))
        self.inference_attention_bias = None
        # relative position index matrix of input
        self.register_buffer('relative_positions',
                             torch.LongTensor(relative_positions).view(resolution, resolution))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        # When in training mode, delete inference_attention_bias
        if mode and hasattr(self, 'inference_attention_bias'):
            del self.inference_attention_bias
        else:
            # Advanced Indexing
            self.inference_attention_bias = self.attention_bias[:, self.relative_positions]

    def forward(self, x):
        # attention_biases will be trained but
        bias = self.attention_bias[:, self.relative_positions]
        B, C, H, W = x.shape
        # B*C/n*H*W
        att_in = x.chunk(self.num_heads, dim=1)
        att_outs = []
        for i in range(self.num_heads):
            if i > 0:
                att_in = att_in + att_outs[i - 1]
            # B*C*H*W
            q = self.dwconv[i](self.q[i](x))
            k = self.k[i](x)
            v = self.v[i](x)
            # BCHW -> BC(HW)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            # BC(HW) * B(HW)C -> B(HW)(HW)
            qk = q.transpose(1, 2) * k * self.sqrtd + (bias[i] if self.training else self.inference_attention_bias[i])
            # B(HW)(HW)
            qk = qk.softmax(dim=-1)
            # BC(HW) * B(HW)(HW) -> BC(HW) ->BCHW
            att_out = (v * qk.transpose(1, 2)).view(B, v.size(1), H, W)
            att_outs.append(att_out)
        x = torch.cat(att_outs, dim=1)
        x = self.act(x)
        x = self.p(x)
        return x


class EfficientVitBlock(nn.Module):
    def __init__(self, channels, qk_dim, v_dim, num_heads, resolution, window_resolution, q_kernel_size, ffn_depth):
        super().__init__()
        self.ffn_depth = ffn_depth
        self.interact1 = [Residual(TokenInteractionBlock(channels, bn_init_weight=0.))] * self.ffn_depth
        self.ffn1 = [Residual(FFN(channels))] * self.ffn_depth
        self.att = LocalWindowAttention(channels, qk_dim, v_dim, num_heads, resolution, window_resolution, q_kernel_size)
        self.interact2 = [Residual(TokenInteractionBlock(channels, bn_init_weight=0.))] * self.ffn_depth
        self.ffn2 = [Residual(FFN(channels))] * self.ffn_depth

    def forward(self, x):
        for i in range(self.ffn_depth):
            x = self.interact1[i](x)
            x = self.ffn1[i](x)
        x = self.att(x)
        for i in range(self.ffn_depth):
            x = self.interact2[i](x)
            x = self.ffn2[i](x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, channels, num_classes):
        super(OutputLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BN_Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class EfficientViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 channels=[64, 128, 192],
                 depth=[1, 2, 3],
                 window_size=[7,7,7],
                 qk_dim=[16, 16, 16],
                 num_heads=[4, 4, 4],
                 ffn_depth=1,
                 q_kernel_size=[[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]],):
        super().__init__()

        for i in range(3):
            assert len(q_kernel_size[i]) == num_heads[i], "q_kernel_size must equal to num_heads."
        # v_dim has same dimension as input_channel in CGA
        v_dim = [(channels[i] / num_heads[i]) for i in range(len(channels))]
        self.patch_emb = OverlapPatchEmbedding(in_channels, channels[0])
        self.evit1 = nn.Sequential()
        self.evit2 = nn.Sequential()
        self.evit3 = nn.Sequential()
        resolution = img_size // patch_size
        for i in range(depth[0]):
            self.evit1.add_module(f"evit1_{i+1}", EfficientVitBlock(channels[0], qk_dim[0], v_dim[0], num_heads[0], resolution, window_size[0], q_kernel_size[0], ffn_depth))
        # subsampling ratio == 2
        resolution = (resolution - 1) // 2 + 1
        for i in range(depth[1]):
            self.evit2.add_module(f"evit2_{i+1}", EfficientVitBlock(channels[1], qk_dim[1], v_dim[1], num_heads[1], resolution, window_size[1], q_kernel_size[1], ffn_depth))
        resolution = (resolution - 1) // 2 + 1
        for i in range(depth[2]):
            self.evit3.add_module(f"evit2_{i+1}", EfficientVitBlock(channels[2], qk_dim[2], v_dim[2], num_heads[2], resolution, window_size[2], q_kernel_size[2], ffn_depth))
        self.subs1 = SubSamplingBlock(channels[0], channels[1], ffn_depth)
        self.subs2 = SubSamplingBlock(channels[1], channels[2], ffn_depth)
        self.output = OutputLayer(channels[2], num_classes)

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.evit1(x)
        x = self.subs1(x)
        x = self.evit2(x)
        x = self.subs2(x)
        x = self.evit3(x)
        x = self.output(x)
        return x
