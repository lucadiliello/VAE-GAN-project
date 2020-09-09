import torch
from torch import nn
from collections import namedtuple

class GANLoss(nn.Module):
    def __init__(self, gan_mode = "lsgan", target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor, device="cuda"):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', "wganr1"]:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            self.real_label_var.requires_grad = False
            target_tensor = self.real_label_var
        else:
            self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            self.fake_label_var.requires_grad = False
            target_tensor = self.fake_label_var
        return target_tensor.to(self.device)

    def forward(self, input, target_is_real):
        if not isinstance(input, list) and input.shape[0] == 0:
            return 0
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                if self.gan_mode in ['lsgan', 'vanilla']:
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
                else:
                    if target_is_real:
                        loss += -torch.mean(pred)
                    else:
                        loss += torch.mean(pred)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
