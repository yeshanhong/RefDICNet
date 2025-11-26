import torch
from RefDCINet import utils
import torch.nn as nn
import torch.nn.functional as F

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(256, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(256, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(256, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(256, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(256, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(256, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        # print(hx.shape)
        z = torch.sigmoid(self.convz1(hx))
        # print(z.shape)
        r = torch.sigmoid(self.convr1(hx))
        # print(r.shape)
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=64, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(64, hidden_dim=128)

    def forward(self, corr, inp, net, flow, upsample=True):

        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)
        # print(net.shape)
        # print(inp.shape)
        # torch.Size([4, 64, 64, 64])
        # torch.Size([4, 192, 64, 64])
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, delta_flow

#####################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

class Refine(torch.nn.Module):
    def __init__(self, context_dim, iter_context_dim, num_layers, levels, radius, inter_dim):
        super(Refine, self).__init__()

        self.radius = radius

        self.conv1 = ConvBlock((radius*2+1)**2*levels+context_dim+iter_context_dim+2+1, context_dim+iter_context_dim, kernel_size=3, stride=1, padding=1)

        self.conv2 = ConvBlock(context_dim+iter_context_dim, inter_dim, kernel_size=3, stride=1, padding=1)

        self.conv_layers = torch.nn.ModuleList([ConvBlock(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1)
                                                for i in range(num_layers)])

        self.conv3 = torch.nn.Conv2d(inter_dim, iter_context_dim+2, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)

        # self.hidden_act = torch.nn.Tanh()
        self.hidden_act = torch.nn.Hardtanh(min_val=-4.0, max_val=4.0)
        # self.hidden_norm = torch.nn.BatchNorm2d(feature_dim)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.radius_emb = torch.tensor(self.radius, dtype=torch.half if amp else torch.float, device=device).view(1,-1,1,1).expand([batch_size,1,height,width])

    def forward(self, corrs, context, iter_context, flow0):

        x = torch.cat([corrs, context, iter_context, flow0, self.radius_emb], dim=1)

        x = self.conv1(x)

        x = self.conv2(x)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.conv3(x)

        return self.hidden_act(x[:,2:]), x[:,:2]
