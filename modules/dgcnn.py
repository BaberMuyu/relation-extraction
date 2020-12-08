import torch
import torch.nn as nn


class DilatedGatedConv1dRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedGatedConv1dRes, self).__init__()
        padding = int(kernel_size / 2) * dilation
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels * 2,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation)

        self.out_channels = out_channels

    def forward(self, seq, mask):
        h = self.conv(seq)
        g, h = h[:, :self.out_channels, :], h[:, self.out_channels:, :]
        g = nn.functional.sigmoid(g)
        seq = seq * (1 - g) + h * g
        seq = seq * mask
        # seq = self.conv(seq)
        return seq


class DGCNN(nn.Module):
    def __init__(self, layer_settings):
        super(DGCNN, self).__init__()
        self.conv_list = []
        for setting in layer_settings:
            layer = DilatedGatedConv1dRes(in_channels=setting['in_channels'],
                                          out_channels=setting['out_channels'],
                                          kernel_size=setting['kernel_size'],
                                          dilation=setting['dilation'])
            self.conv_list.append(layer)
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, seq, mask):
        seq = seq.permute(0, 2, 1).contiguous()
        mask = mask.unsqueeze(dim=1)
        for layer in self.conv_list:
            seq = layer(seq, mask)
        seq = seq.permute(0, 2, 1)
        return seq

# def dilated_gated_conv1d(seq, mask, dilation_rate=1):
#     """膨胀门卷积（残差式）
#     """
#     dim = K.int_shape(seq)[-1]
#     h = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(seq)
#
#     def _gate(x):
#         dropout_rate = 0.1
#         s, h = x
#         g, h = h[:, :, :dim], h[:, :, dim:]
#         g = K.in_train_phase(K.dropout(g, dropout_rate), g)
#         g = K.sigmoid(g)
#         return g * s + (1 - g) * h
#
#     seq = Lambda(_gate)([seq, h])
#     seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
#     return seq
