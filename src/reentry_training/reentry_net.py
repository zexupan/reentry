import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apex import amp
import copy
from speaker_encoder_net import SpeakerEncoder
from slsyn_net import slsyn_net

EPS = 1e-8

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class reentry_net(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, M, V, pretrained_v=1):
        super(reentry_net, self).__init__()
        self.pretrained_v = pretrained_v

        # audio encoder and bottleneck layer
        self.a_encoder = audioEncoder(L, N)
        self.a_norm =nn.Sequential(ChannelWiseLayerNorm(N), nn.Conv1d(N, B, 1, bias=False))

        # video encoder and adaption layer for each tcn
        self.v_encoder = videoEncoder(V = V)
        self.v_adapt_mask = _clones(nn.Conv1d(V, B, 1, bias=False),R)
        # self.v_adapt_spk = _clones(nn.Conv1d(V, B, 1, bias=False),R)

        # the tenporal cnn blocks
        self.projection_0 = nn.Conv1d(B*2, B, 1, bias=False)
        self.projection = _clones(nn.Conv1d(B*3, B, 1, bias=False), R-1)
        self.tcn = _clones(TCN_block(X,P,B,H), R)

        # speaker embedding extraction and classification
        self.spk_encoder=_clones(SpeakerEncoder(B), R-1)
        self.spk_classifier=_clones(nn.Linear(B,M), R-1)

        # Mask generation layer
        self.a_decoder = audioDecoder(B, N, L)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # the pretrained models
        self.v_front_end = slsyn_net()
        pretrained_model = torch.load("../../pretrained_slsyn/slsyn_model_dict.pt")['model']
        state = self.v_front_end.state_dict()
        for key in state.keys():
            pretrain_key = 'module.' + key
            state[key] = pretrained_model[pretrain_key]
        self.v_front_end.load_state_dict(state)
        if self.pretrained_v:
            for param in self.v_front_end.parameters():
                param.requires_grad = False

        
        # self.v_ds = nn.Linear(40,256, bias=False)


    def forward(self, mixture, v):
        if not self.pretrained_v:
            v = self.v_front_end(mixture, v).transpose(1,2)

        T_origin = mixture.size(-1)

        # audio encoder
        mixture_w = self.a_encoder(mixture)
        est_mask = self.a_norm(mixture_w)

        # video encoder
        # v = self.v_ds(v)
        v = self.v_encoder(v)

        spks=[]
        # tcn blocks
        for i in range(len(self.tcn)):
            v_mask = self.v_adapt_mask[i](v)
            v_mask = F.interpolate(v_mask, (32*v_mask.size(2)), mode='linear')
            v_mask = F.pad(v_mask,(0,est_mask.size(2)-v_mask.size()[2]))

            if i==0:
                est_mask = torch.cat((est_mask, v_mask),1)
                est_mask = self.projection_0(est_mask)
                est_mask = self.tcn[i](est_mask)
            else:
                # int_speech = mixture_w * F.relu(est_mask)
                # spk_emb, spk_emb_avg = self.spk_encoder[i-1](int_speech)

                spk_emb, spk_emb_avg = self.spk_encoder[i-1](self.a_encoder(self.a_decoder(mixture_w, est_mask, T_origin)))
                
                if i <= 3:
                    spk_emb = torch.repeat_interleave(spk_emb_avg.unsqueeze(2), repeats=est_mask.size(2), dim=2)
                spks.append(self.spk_classifier[i-1](spk_emb_avg))

                est_mask = torch.cat((spk_emb, est_mask, v_mask),1)
                est_mask = self.projection[i-1](est_mask)
                est_mask = self.tcn[i](est_mask)

        # decoder
        est_source = self.a_decoder(mixture_w, est_mask, T_origin)

        # return spk_emb_avg.squeeze(), est_source
        return spks, est_source

class audioEncoder(nn.Module):
    def __init__(self, L, N):
        super(audioEncoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.conv1d_U(x))
        return x

class audioDecoder(nn.Module):
    def __init__(self, B, N, L):
        super(audioDecoder, self).__init__()
        self.N, self.L = N, L
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)
        # self.de_conv1d_U = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L // 2, bias=False)
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask, T_origin):
        est_mask = self.mask_conv1x1(est_mask)
        x = mixture_w * F.relu(est_mask)  # [M,  N, K]
        # est_source = self.de_conv1d_U(x).squeeze(1)
        x = torch.transpose(x, 2, 1) # [M,  K, N]
        x = self.basis_signals(x)  # [M,  K, L]
        est_source = overlap_and_add(x, self.L//2) # M x C x T

        # T changed after conv1d in encoder, fix it here
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

class videoEncoder(nn.Module):
    def __init__(self,V=256,R=5):
        super(videoEncoder, self).__init__()
        ve_blocks = []
        for x in range(R):
            ve_blocks +=[VisualConv1D(V)]
        self.net = nn.Sequential(*ve_blocks)

    def forward(self, v):
        v = v.transpose(1,2)
        v = self.net(v)
        return v

class VisualConv1D(nn.Module):
    def __init__(self, V, H=512):
        super(VisualConv1D, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(V)
        conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(H)
        dsconv = nn.Conv1d(H, H, 3, stride=1, padding=1,dilation=1, groups=H, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(H)
        pw_conv = nn.Conv1d(H, V, 1, bias=False)
        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

class TCN_block(nn.Module):
    def __init__(self, X, P, B, H):
        super(TCN_block, self).__init__()
        tcn_blocks = []
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            tcn_blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]
        self.tcn = nn.Sequential(*tcn_blocks)

    def forward(self, x):
        x = self.tcn(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)

    def forward(self, x):
        return self.net(x)

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result
