import torch
import torch.nn as nn
from torch import flatten
# from torchinfo import summary

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = nn.Conv3d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                    kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm3d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, d, h, w = x.size()
        fft_dim = (-3, -2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (batch, c, d, h, w/2+1, 2)
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()   # (batch, c, 2, d, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))                      # (batch, 2*c, d, h, w/2+1)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])\
            .permute(0, 1, 3, 4, 5, 2).contiguous()            # (batch, c, d, h, w/2+1, 2)

        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice,
                                  dim=fft_dim, norm='ortho')   # (batch, c, d, h, w)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=(1, 1, 1), groups=groups, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=(1, 1, 1), groups=1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)


        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type="reflect", **spectral_kwargs):
        super(FFC, self).__init__()

        self.stride = stride
        in_cg = int(in_channels * ratio_gin)  # number of input channels for global part
        in_cl = in_channels - in_cg  # number of input channels for local part
        out_cg = int(out_channels * ratio_gout)  # number of output channels for global part
        out_cl = out_channels - out_cg  # number of output channels for local part

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l_module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv3d
        self.convl2l = self.convl2l_module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        self.convl2g_module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv3d
        self.convl2g = self.convl2g_module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        self.convg2l_module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv3d
        self.convg2l = self.convg2l_module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        self.convg2g_module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = self.convg2g_module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        # if type(x) is tuple:
        #     print(x_l.shape, x_g.shape)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            if self.convl2l_module is nn.Identity:
                out_xl = self.convg2l(x_g)
            elif self.convg2l_module is nn.Identity:
                out_xl = self.convl2l(x_l)
            else:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            if self.convl2g_module is nn.Identity:
                out_xg = self.convg2g(x_g)
            elif self.convg2g_module is nn.Identity:
                out_xg = self.convl2g(x_l)
            else:
                out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm3d, activation_layer=nn.ReLU(True),
                 padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)

        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact
        self.act_g = gact

        # self.maxpool = nn.MaxPool3d(kernel_size=pooling_size, stride=pooling_stride)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFC3D(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm3d, padding_type='reflect',
                 activation_layer=nn.ReLU(True), is_training=True,
                 init_conv_kwargs={}, internal_conv_kwargs={}, normal_conv_kwargs={}):
        super(FFC3D, self).__init__()
        self.is_training = is_training

        # self.conv1_1 = nn.Conv3d(input_nc, 16, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=0)                           
        # self.conv1_2 = nn.Conv3d(16, 16, kernel_size=(2, 13, 13), stride=(1, 1, 1), padding=0)
        # self.conv1_3 = nn.Conv3d(16, 64, kernel_size=(1, 10, 10), stride=(1, 1, 1), padding=0)
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1)
        # self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=1)
        # self.relu1 = nn.ReLU(True)
        # self.relu2 = nn.ReLU(True)
        # self.relu3 = nn.ReLU(True)
        # self.batchnorm1 = nn.BatchNorm3d(16)
        # self.batchnorm2 = nn.BatchNorm3d(16)
        # self.batchnorm3 = nn.BatchNorm3d(64)

        self.ffc_conv1 = FFC_BN_ACT(input_nc, 16, kernel_size=(1, 7, 7),
                                     padding=0, norm_layer=norm_layer, stride=2,
                                     activation_layer=activation_layer, **init_conv_kwargs)

        self.ffc_conv2 = FFC_BN_ACT(16, 16, kernel_size=(2, 13, 13),
                                     padding=(0, 5, 5), norm_layer=norm_layer, stride=2,
                                     activation_layer=activation_layer, **internal_conv_kwargs)

        self.ffc_conv3 = FFC_BN_ACT(16, 64, kernel_size=(1, 11, 11),
                                     padding=(0, 5, 5), norm_layer=norm_layer, stride=1,
                                     activation_layer=activation_layer, **internal_conv_kwargs)

        self.fc1 = nn.Linear(in_features=57600, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # x = self.conv1_1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)
        # x = self.batchnorm1(x)

        # print(x.shape)

        # x = self.conv1_2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        # x = self.batchnorm2(x)

        # print(x.shape)

        # x = self.conv1_3(x)
        # x = self.relu3(x)
        # x = self.maxpool3(x)
        # x = self.batchnorm3(x)

        x = self.ffc_conv1(x)
        # print(x[0].shape, x[1].shape)      
        x = self.ffc_conv2(x)
        # print(x[0].shape, x[1].shape)
        x = self.ffc_conv3(x)
        # print(x[0].shape, x[1].shape)
        x = torch.cat((x[0], x[1]), dim=1)
        x = flatten(x, 1)
        x = self.fc1(x)
        if self.is_training:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.is_training:
            x = self.dropout(x)
        output = self.fc3(x)
        # if self.is_training:
        #     x = self.dropout(x)
        # output = self.fc4(x)

        # output = torch.sigmoid(x)
        # print(output)
        return output


# init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0.5, 'enable_lfu': False}
# internal_conv_kwargs = {'ratio_gin': 0.5, 'ratio_gout': 0.5, 'enable_lfu': False}
# model = FFC3D(input_nc=1, init_conv_kwargs=init_conv_kwargs, internal_conv_kwargs=internal_conv_kwargs)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# # print(summary(model, input_size=[10, 1, 3, 128, 128], mode="train", device='cpu'))

# tensor = torch.zeros([10, 1, 3, 128, 128], dtype=torch.float32)
# init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0.75, 'enable_lfu': False}
# internal_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
# model = FFC3D(input_nc=1, init_conv_kwargs=init_conv_kwargs, internal_conv_kwargs=internal_conv_kwargs)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# out = model(tensor.to(device))
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of total parameteres: {}".format(pytorch_total_params))