# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
import kornia

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    # if opt.lr_policy == 'lambda':
    lambda_rule = lambda epoch: 0.2 ** ((epoch + 1) // 5)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # elif opt.lr_policy == 'step':
    # scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    # else:
    #	return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    net = net

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'pretrained':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids).cuda()

    for root_child in net.children():
        for children in root_child.children():
            if children in root_child.need_initialization:
                init_weights(children, init_type, gain=init_gain)
    return net


def define_DCCASparseNet(rgb_enc=True, depth_enc=True, depth_dec=True, norm='batch', init_type='xavier', init_gain=0.02,
                         gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = DCCASparsenetGenerator(rgb_enc=rgb_enc, depth_enc=depth_enc, depth_dec=depth_dec)
    return init_net(net, init_type, init_gain, gpu_ids)


class SAConv(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
		super(SAConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		x = x * m
		x = self.conv(x)
		weights = torch.ones(torch.Size([1, 1, 3, 3])).float().cuda()
		#weights = torch.ones(torch.Size([1, 1, 3, 3]))
		mc = F.conv2d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc * 9

		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = self.pool(m)

		return x, m

class SAConv2(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
		super(SAConv2, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		x = x * m
		x = self.conv(x)
		weights = torch.ones(torch.Size([1, 1, 3, 3])).float().cuda()
		#weights = torch.ones(torch.Size([1, 1, 3, 3]))
		mc = F.conv2d(m, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc * 9

		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		#m = self.pool(m)

		return x, m



class SAConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(SAConvBlock, self).__init__()
        self.sparse_conv = SAConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x, m = input
        x, m = self.sparse_conv((x, m))
        assert (m.size(1) == 1)
        x = self.relu(x)

        return x, m

def make_blocks_from_names(names, in_dim, out_dim):
    layers = []
    if names[0] == "block1" or names[0] == "block2":
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    else:
        layers += [SAConvBlock(in_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
        layers += [SAConvBlock(out_dim, out_dim, 3, stride=1)]
    return nn.Sequential(*layers)

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        #self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.conv = SAConv(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #self.pool.require_grad = False

    def forward(self, input):
        x, m = input
        #x = x * m
        #print("x shape", x.shape)
        x_1, m = self.conv((x,m))
        output = torch.cat([x_1, self.pool(x)], 1)
        #print("cat shape", output.shape)
        output = self.bn(output)
        return F.relu(output), m


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = SAConv(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = SAConv(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = SAConv2(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = SAConv2(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self,input):
        x,m = input

        output, m = self.conv3x1_1((x,m))
        output = F.relu(output)
        output,m = self.conv1x3_1((output,m))
        output = self.bn1(output)
        output = F.relu(output)
        #print("conv3_1 shape", output.shape)
        #print("corresponding mask shape", m.shape)
        output,m = self.conv3x1_2((output,m))
        output = F.relu(output)
        output, m = self.conv1x3_2((output,m))
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + x), m # +input = identity (residual connection)


class non_bottleneck_1d_dec(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        # only for encoder mode:
        #self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output, m = self.initial_block(input)

        for layer in self.layers:
            output, m = layer((output,m))

        #if predict:
        #   output = self.output_conv(output)

        return output, m


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d_dec(64, 0, 1))
        self.layers.append(non_bottleneck_1d_dec(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 32))
        self.layers.append(non_bottleneck_1d_dec(32, 0, 1))
        self.layers.append(non_bottleneck_1d_dec(32, 0, 1))

        #self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        features = 32
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x

class iAFF(nn.Module):
    """
    Reference: https://github.com/YimianDai/open-aff

    Alternative Fusion Module to fuse depth and rgb features
    """

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class DCCASparsenetGenerator(nn.Module):
    def __init__(self, rgb_enc=True, depth_enc=True, depth_dec=True):  # use encoder to pass pretrained encoder
        super().__init__()
        num_classes = 64
        self.need_initialization = []


        if rgb_enc:
            self.encoder_rgb = Encoder(3,num_classes)
        if depth_enc:
            self.encoder_depth = Encoder(1,num_classes)
        if depth_dec:
            self.decoder = Decoder(64)
        self.Transform = make_blocks_from_names(["block1"], 128, 128)
        self.fusion = iAFF(channels=128)
        features = 32
        non_negative = True

        # ## output layer changed 03012020
        # self.output_conv = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(False),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        #     nn.Identity(),
        # )
        #self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.need_initialization.append(self.decoder)
        #self.need_initialization.append(self.conv3)

    def forward(self, sparse_rgb, sparse_d, mask, rgb, d):
        
        x_dataview, m_d = self.encoder_depth((sparse_d, mask))


        ##rgb
        y_dataview,m_r = self.encoder_rgb((sparse_rgb, mask))

        ##sparse rgb##
        inverse_mask = torch.ones_like(mask) - mask
        inverse_rgb = rgb * inverse_mask
        ym_dataview, m_m = self.encoder_rgb((inverse_rgb, inverse_mask))

        x_trans, m_trans = self.Transform((y_dataview, m_r))
        xm_trans, mm_trans = self.Transform((ym_dataview, m_r))
        tst = torch.cat((x_dataview, xm_trans),1)

        fusedinp = self.fusion(xm_trans, x_dataview)
        depth_est = self.decoder(fusedinp)

        return x_dataview, y_dataview, x_trans, depth_est

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # valid_mask = (target>0).detach()
        valid_mask = (target > 0) & (target <=500).detach()

        #diff = target - pred
        #diff = diff[valid_mask]
        #pred[pred <= 0] = 0.001
        #target[target <= 0] = 0.001
        diff = target - pred
        diff = diff[valid_mask]
        #pgrads: torch.Tensor = kornia.spatial_gradient(torch.log(pred), order=1)
       # tgrads: torch.Tensor = kornia.spatial_gradient(torch.log(target), order=1)
        #diff2 = torch.sqrt(tgrads[:, :, 0] ** 2 + tgrads[:, :, 1] ** 2) - torch.sqrt(pgrads[:, :, 0] ** 2 + pgrads[:, :, 1] ** 2)
        self.loss = (diff ** 2).mean()
                    #+ 0.1 * (diff2 ** 2).mean()
        return self.loss

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask[mask==True])
    valid = ssum > 0
    m = torch.zeros_like(mask[mask==True])
    s = torch.ones_like(mask[mask==True])

    m = torch.median((mask[valid] * target[valid]).view(valid.sum(), -1), dim=1).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask[valid] * target[valid].abs())
    s = torch.clamp((sq / ssum), min=1e-6)

    return target / (s.view(-1, 1, 1))
class TrMAELoss(nn.Module):
    def __init__(self):
        super(TrMAELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        mask = (target>0).detach()
        #mask = (target > 0) & (target <= 500).detach()
        #pred = normalize_prediction_robust(pred,mask)
        #target = normalize_prediction_robust(target,mask)

        res = pred - target
        #res = res[mask.bool()].abs().mean()
        res = (res[mask.bool()] ** 2).mean()
        M = torch.sum(mask)
        self.loss = res
        gradl = 0
        #pred = normalize_prediction_robust(pred,mask)
        #target = normalize_prediction_robust(target,mask)
        for scale in range(1, 5):
            pred_r = F.interpolate(pred, scale_factor=1/scale, mode='bilinear')
            target_r = F.interpolate(target,scale_factor=1/scale, mode='bilinear')
            mask = (target_r > 0).detach()
            pgrads: torch.Tensor = kornia.spatial_gradient(pred_r, order=1)
            tgrads: torch.Tensor = kornia.spatial_gradient(target_r, order=1)
            #res = pred_r - target_r
            #trim = 0.2
            diff2 = (pgrads[:, :, 0] - tgrads[:, :, 0]) + (pgrads[:, :, 1]  - tgrads[:, :, 1])
            #res = res[mask.bool()].abs().mean()
            #diff2 = diff2[mask.bool()].abs().mean()
            gradl+= diff2.abs().mean()

       # trimmed, _ = torch.sort(res.view(-1), descending=False)[
        #             : int(len(res) * (1.0 - trim))
        #             ]

        #self.loss = trimmed.sum() / (2 * M.sum()) + diff2
        self.loss+= 0.5 * gradl
        return self.loss


# class EdgeMaskedMSELoss(nn.Module):
#     def __init__(self):
#         super(MaskedMSELoss, self).__init__()
#
#     def forward(self, pred, target):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         # valid_mask = (target>0).detach()
#         #valid_mask = (target > 0) & (target <=500).detach()
#         edgest = kornia.filters.Sobel()(target)
#         edgesp = kornia.filters.Sobel()(pred)
#         diff = edgest - edgesp
#         #diff = diff[valid_mask]
#         self.loss = (diff ** 2).mean()
#         return self.loss

class TransformLoss(nn.Module):
    def __init__(self):
        super(TransformLoss, self).__init__()

    def forward(self, f_in, f_target):
        assert f_in.dim() == f_target.dim(), "inconsistent dimensions"
        diff = f_in - f_target
        self.loss = (diff ** 2).mean()
        return self.loss


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, pred_map):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1.

        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
            weight /= 2.3  # don't ask me why it works better
        return loss


class DCCA_2D_Loss(nn.Module):
    def __init__(self, outdim_size, use_all_singular_values, device):
        super(DCCA_2D_Loss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def __call__(self, data_view1, data_view2):
        H1 = data_view1.view(data_view1.size(0) * data_view1.size(1), data_view1.size(2), data_view1.size(3))
        H2 = data_view2.view(data_view2.size(0) * data_view2.size(1), data_view2.size(2), data_view2.size(3))

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        corr_sum = 0
        o1 = o2 = H1.size(1)

        m = H1.size(0)
        n = H1.size(1)

        H1bar = H1 - (1.0 / m) * H1
        H2bar = H2 - (1.0 / m) * H2
        Hat12 = torch.zeros(m, n, n).cuda()
        Hat11 = torch.zeros(m, n, n).cuda()
        Hat22 = torch.zeros(m, n, n).cuda()
        # Hat12 = torch.zeros(m,n,n)
        # Hat11 = torch.zeros(m,n,n)
        # Hat22 = torch.zeros(m,n,n)

        for i in range(m):
            Hat11[i] = torch.matmul(H1bar[i], H1bar.transpose(1, 2)[i])
            Hat12[i] = torch.matmul(H1bar[i], H2bar.transpose(1, 2)[i])
            Hat22[i] = torch.matmul(H2bar[i], H2bar.transpose(1, 2)[i])

        SigmaHat12 = (1.0 / (m - 1)) * torch.mean(Hat12, dim=0)
        SigmaHat11 = (1.0 / (m - 1)) * torch.mean(Hat11, dim=0) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.mean(Hat22, dim=0) + r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = torch.sqrt(torch.trace(torch.matmul(Tval.t(), Tval)))
        else:
            # just the top self.outdim_size singular values are used
            U, V = torch.symeig(torch.matmul(Tval.t(), Tval), eigenvectors=True)
            U = U[torch.gt(U, eps).nonzero()[:, 0]]
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def __call__(self, x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
        pool2d = nn.AvgPool2d(kernel_size, stride=stride)
        refl = nn.ReflectionPad2d(1)

        x, y = refl(x), refl(y)
        mu_x = pool2d(x)
        mu_y = pool2d(y)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = pool2d(x.pow(2)) - mu_x_sq
        sigma_y = pool2d(y.pow(2)) - mu_y_sq
        sigma_xy = pool2d(x * y) - mu_x_mu_y
        v1 = 2 * sigma_xy + C2
        v2 = sigma_x + sigma_y + C2

        ssim_n = (2 * mu_x_mu_y + C1) * v1
        ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
        ssim = ssim_n / ssim_d
        ssim = torch.clamp((1. - ssim) / 2., 0., 1.)

        return ssim