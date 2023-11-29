import torch as th
import torch.nn as nn
import torch.nn.functional as func
import functools


# Descriminator Class. Initialised with zero weights. Layers as decribed in the code.
class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        scn,
        ccn,
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        use_proj=True,
        use_cnt=False,
    ):
        super(PatchDiscriminator, self).__init__()
        self.use_proj = use_proj
        self.use_cnt = use_cnt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # kernel size
        padw = 1  # padding
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*sequence)

        sequence = []
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.bclass = nn.Sequential(*sequence)
        if use_proj:
            self.sproj = nn.Embedding(scn, ndf * nf_mult)
            self.sproj.weight.data.fill_(0)  # zero weight initialization
            if self.use_cnt:
                self.cproj = nn.Embedding(ccn, ndf * nf_mult)
                self.cproj.weight.data.fill_(0)  # zero weight initialization

        sequence = [nn.Linear(512 * 3 * 3, scn)]
        self.sclass = nn.Sequential(*sequence)

        if self.use_cnt:
            sequence = [nn.Linear(512 * 3 * 3, ccn)]
            self.cclass = nn.Sequential(*sequence)

    # forward pass on Discriminator
    def forward(self, indata, slabel=None, clabel=None):
        ftr = self.model(indata)
        bc_out = self.bclass(ftr)  # True/False
        if self.use_proj:  # projection discriminator
            if slabel is not None:
                semb = self.sproj(slabel)
                sftr = ftr * semb.view(semb.size(0), semb.size(1), 1, 1)
                bc_out += th.mean(sftr, dim=1, keepdim=True)
            if self.use_cnt and clabel is not None:
                cemb = self.cproj(clabel)
                cftr = ftr * cemb.view(cemb.size(0), cemb.size(1), 1, 1)
                bc_out += th.mean(cftr, dim=1, keepdim=True)

        ftr = func.avg_pool2d(ftr, 9)  # average pooling
        ftr = ftr.view(ftr.size(0), -1)  # reshape
        sc_out = self.sclass(ftr)  # Style classificatin
        if self.use_cnt:
            cc_out = self.cclass(ftr)  # content classificatin
        else:
            cc_out = None

        return bc_out, cc_out, sc_out

    # loading a trained model for testing
    def load_model(self, load_model):
        checkpoint = th.load(load_model)
        self.load_state_dict(checkpoint["disc"])
        print("discriminator loaded from:", load_model)
