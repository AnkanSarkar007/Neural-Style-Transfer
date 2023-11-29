import torch as th
from torch.autograd import Variable
import torch.nn as nn


class gan_style_loss(nn.Module):
    def __init__(
        self, real_lbl=1.0, fake_lbl=0.0, use_cuda=True
    ):  # initialize the various kinds of losses
        super(gan_style_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.use_cuda = use_cuda

        self.real_lbl = real_lbl
        self.fake_lbl = fake_lbl
        self.real_lbl_var = None
        self.fake_lbl_var = None
        self.ls = []
        self.pls = []
        self.use_p = False

    def parallel(self, gids):
        self.mse = nn.DataParallel(self.mse, device_ids=gids)
        self.l1 = nn.DataParallel(self.l1, device_ids=gids)
        self.use_p = True

    def reset(self):
        self.ls = []
        self.pls = []  # parallel loss

    # MSE loss
    def add_mse(self, input, target, weight=1.0):
        l = weight * self.mse(input, target)
        if self.use_p:
            self.pls.append(l)
        else:
            self.ls.append(l)

    # L1 loss
    def add_l1(self, input, target, weight=1.0):
        l = weight * self.l1(input, target)
        if self.use_p:
            self.pls.append(l)
        else:
            self.ls.append(l)

    # Cross-Entropy Loss
    def add_ce(self, input, target, weight=1.0):
        l = weight * self.ce(input, target)
        self.ls.append(l)

    # Getting the label of style image
    def get_real_lbl_var(self, indata):
        if self.real_lbl_var is None or (self.real_lbl_var.numel() != indata.numel()):
            self.real_lbl_var = th.FloatTensor(indata.size()).fill_(self.real_lbl)
            if self.use_cuda:
                self.real_lbl_var = self.real_lbl_var.cuda()
            self.real_lbl_var = Variable(self.real_lbl_var, requires_grad=False)

    # Getting the label of reconstructed image
    def get_fake_lbl_var(self, indata):
        if self.fake_lbl_var is None or (self.fake_lbl_var.numel() != indata.numel()):
            self.fake_lbl_var = th.FloatTensor(indata.size()).fill_(self.fake_lbl)
            if self.use_cuda:
                self.fake_lbl_var = self.fake_lbl_var.cuda()
            self.fake_lbl_var = Variable(self.fake_lbl_var, requires_grad=False)

    # Probability of determining whether image is real
    def add_gan_real(self, input, weight=1.0):
        self.get_real_lbl_var(input)
        l = weight * self.bce(input, self.real_lbl_var)
        self.ls.append(l)

    # Probability of determining whether image is fake
    def add_gan_fake(self, input, weight=1.0):
        self.get_fake_lbl_var(input)
        l = weight * self.bce(input, self.fake_lbl_var)
        self.ls.append(l)

    # Final loss is just the sum of all individual losses
    def return_loss(self):
        return sum(self.ls)

    def return_ploss(self):
        return sum(self.pls)
