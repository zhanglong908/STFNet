import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SpatioTemporalSE_Hierarchical(nn.Module):
    def __init__(self, channel, layer_depth):
        super().__init__()

        mid_channels = channel // 4
        kernel_size = (3, 1, 1)

        if layer_depth == 'layer2':
            groups = channel // 4
        elif layer_depth == 'layer3':
            groups = channel // 8
        else:
            groups = channel // 16

        groups = max(1, groups)

        self.temporal_att = nn.Sequential(
            nn.Conv3d(channel, mid_channels, kernel_size=kernel_size,
                      padding=(1, 0, 0), groups=groups, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, channel, kernel_size=kernel_size,
                      padding=(1, 0, 0), groups=groups, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        temporal_att = self.temporal_att(x)
        return x * temporal_att.expand_as(x)


class CSA_NoAlignment(nn.Module):
    def __init__(self, inplanes, planes=1024):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 512, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

        self.stf1 = SpatioTemporalSE_Hierarchical(512, layer_depth='layer2')
        self.stf2 = SpatioTemporalSE_Hierarchical(512, layer_depth='layer3')
        self.stf3 = SpatioTemporalSE_Hierarchical(512, layer_depth='layer4')

        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d((None, 7, 7)),
            nn.AdaptiveAvgPool3d((None, 7, 7)),
            nn.Identity()
        ])

        self.tcn_layers = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(planes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 1000)

    def forward(self, res):
        B, _, T, _, _ = res[1].shape
        multi = []

        multi.append(self.stf1(self.conv2(res[1])))
        multi.append(self.stf2(self.conv3(res[2])))
        multi.append(self.stf3(res[3]))

        for i in range(3):
            multi[i] = self.adaptive_pools[i](multi[i])

            multi[i] = multi[i].transpose(1, 2).contiguous()
            multi[i] = multi[i].view((-1,) + multi[i].size()[2:])

            if i < 2:
                multi[i] = F.adaptive_avg_pool2d(multi[i], (1, 1))
            else:
                multi[i] = self.avgpool(multi[i])

            multi[i] = multi[i].view(multi[i].size(0), -1)
            multi[i] = self.fc(multi[i])
            multi[i] = multi[i].view(B, T, -1).permute(0, 2, 1)
            multi[i] = self.tcn_layers(multi[i])

        return multi


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        self.csa = CSA_NoAlignment(1000)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            framewise, res = self.conv2d(x.permute(0, 2, 1, 3, 4))
            framewise = framewise.view(batch, temp, -1).permute(0, 2, 1)
        else:
            framewise = x

        res2 = self.csa(res)
        for i in range(3):
            res2[i] = res2[i].permute(2, 0, 1)
            res2[i] = self.classifier(res2[i])

        conv1d_outputs = self.conv1d(framewise, len_x)
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "res2": res2,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()

                loss += self.loss['CTCLoss'](ret_dict["res2"][0].log_softmax(-1),
                                             label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                             label_lgt.cpu().int()).mean()
                loss += self.loss['CTCLoss'](ret_dict["res2"][1].log_softmax(-1),
                                             label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                             label_lgt.cpu().int()).mean()
                loss += self.loss['CTCLoss'](ret_dict["res2"][2].log_softmax(-1),
                                             label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                             label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'Cu':
                loss += weight * ret_dict["loss_LiftPool_u"]
            elif k == 'Cp':
                loss += weight * ret_dict["loss_LiftPool_p"]
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['mse_loss'] = nn.MSELoss()
        return self.loss