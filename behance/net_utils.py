import torch as th
import torch.nn as nn

# encoder configurations of individual blocks
cfg = {
        5: [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512],#vgg19, block 5, 14 cnvs
        4: [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512],#vgg19, block 4
        3: [64, 'M', 128, 128, 'M', 256],#vgg19, block 3
        2: [64, 'M', 128],#vgg19, block 2
        1: [64],#vgg19, block 1
        }

# decoder configuration (symmetrical)
dec_cfg = {
    5: [512, "M", 512, 512, 512, "M", 256, 256, 256, "M", 128, 128, "M", 64],
    4: [512, "M", 256, 256, 256, "M", 128, 128, "M", 64],
    3: [256, "M", 128, 128, "M", 64],
    2: [128, "M", 64],
    1: [64],
}

# layers to pick from VGG
th_cfg = {
    5: [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24],
    4: [0, 2, 5, 7, 10, 12, 14, 17],
    3: [0, 2, 5, 7, 10],
    2: [0, 2, 5],
    1: [0, 2],
}

# never required as decoder is trained from scratch
th_dec_cfg = {
    5: [1, 5, 8, 11, 14, 18, 21, 24, 27, 31, 34, 38, 41],
    4: [1, 5, 8, 11, 14, 18, 21, 25, 28],
    3: [1, 5, 8, 12, 15],
    2: [1, 5, 8],
    1: [1],
}


# forming the first encoder block
def make_wct_enc_layers(cfg):
    layers = [nn.Conv2d(3, 3, kernel_size=1, padding=0)]
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            layers += [nn.ReflectionPad2d((2, 2, 2, 2)), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# forming the subsequent encoder blocks (code optimization)
def make_wct_aux_enc_layers(cfg, aux_cfg):
    assert len(cfg) < len(aux_cfg)
    layers = []
    i = 0
    in_channels = None
    while i < len(cfg):
        assert cfg[i] == aux_cfg[i]
        v = cfg[i]
        if v != "M":
            in_channels = v
        i += 1
    while i < len(aux_cfg):
        v = aux_cfg[i]
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            layers += [nn.ReflectionPad2d((1, 1, 1, 1)), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


# forming the trainable decoder layers
def make_tr_dec_layers(
    cfg, in_channels=0, use_bn="b", use_sgm="sigmoid"
):  # trainable decoder
    if in_channels < 1:
        in_channels = cfg[0]
    layers = [
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=0),
        nn.LeakyReLU(0.2, True),
    ]  # first layer without BN
    in_channels = cfg[0]
    i = 1
    while i < len(cfg):
        v = cfg[i]
        if use_bn == "in":
            layers += [nn.InstanceNorm2d(in_channels, affine=True)]
        elif use_bn == "b":
            layers += [nn.BatchNorm2d(in_channels)]
        else:
            print("make_tr_dec: unknown bn")
        if v == "M":
            i += 1
            v = cfg[i]
            conv2d = nn.ConvTranspose2d(
                in_channels, v, kernel_size=4, stride=2, padding=1, bias=(not use_bn)
            )
            layers += [conv2d, nn.LeakyReLU(0.2, True)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=0, bias=(not use_bn)
            )
            layers += [
                nn.ReflectionPad2d((1, 1, 1, 1)),
                conv2d,
                nn.LeakyReLU(0.2, True),
            ]
        in_channels = v
        i += 1

    layers += [
        nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)
    ]  # last layer, create image
    if use_sgm == "sigmoid":  # constrained the pixel value to be 0~1
        layers += [nn.Sigmoid()]
    elif use_sgm == "tanh":
        layers += [nn.Tanh()]
    elif use_sgm == "hard":
        layers += [nn.Hardtanh(min_val=0)]
    elif use_sgm.lower() != "none":
        print(("unknow last decoder layer flag:", use_sgm))
    return nn.Sequential(*layers)


# making the discriminator layers (similar to decoder func)
def make_dise_layers(
    in_channels, out_channels, layer_cfg, use_bn="in", dropout=0.5, use_sgm="none"
):
    # layers= []
    v = int(layer_cfg[0])
    layers = [
        nn.Conv2d(in_channels, v, kernel_size=1, padding=0),
        nn.LeakyReLU(0.2, True),
    ]  # first layer without BN
    in_channels = v
    i = 0
    while i < len(layer_cfg):
        v = layer_cfg[i]
        if use_bn == "in":
            layers += [nn.InstanceNorm2d(in_channels, affine=True)]
        elif use_bn == "b":
            layers += [nn.BatchNorm2d(in_channels)]
        else:
            print("make_dise: unknown bn")
        if v == "D":  # downsample
            i += 1
            v = int(layer_cfg[i])
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=4, stride=2, padding=0, bias=(not use_bn)
            )
            layers += [conv2d, nn.LeakyReLU(0.2, True)]
        elif v == "U":  # upsample
            i += 1
            v = int(layer_cfg[i])
            conv2d = nn.ConvTranspose2d(
                in_channels, v, kernel_size=4, stride=2, padding=0, bias=(not use_bn)
            )
            layers += [conv2d, nn.LeakyReLU(0.2, True)]
        elif v.isdigit():
            v = int(v)
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=0, bias=(not use_bn)
            )
            layers += [
                nn.ReflectionPad2d((1, 1, 1, 1)),
                conv2d,
                nn.LeakyReLU(0.2, True),
            ]
        else:
            print(("make_dise_layers: unknown layer flag", v))
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        in_channels = v
        i += 1
    layers += [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    ]  # last layer to resize
    if use_sgm == "sigmoid":  # constrained the pixel value to be 0~1
        layers += [nn.Sigmoid()]
    elif use_sgm == "tanh":
        layers += [nn.Tanh()]
    elif use_sgm == "hard":
        layers += [nn.Hardtanh(min_val=0)]
    elif use_sgm.lower() != "none":
        print(("unknow last decoder layer flag:", use_sgm))
    return nn.Sequential(*layers)


# retrieves the gram matrix given an image
def get_gram(ftr, use_norm=True):
    # ftr = func.avg_pool2d(ftr, kernel_size=2, stride=2)  #pooling to make receptive field larger
    a, b, c, d = ftr.size()  # a=batch size(=1)
    features = ftr.view(a, b, c * d)  # resise F_XL into \hat F_XL
    G = th.bmm(features, features.transpose(1, 2))  # compute the gram product
    if use_norm:
        return G.div(b * c * d)
    else:
        return G.div(b)


def adin_transform2(base, base2):  # AdaIN transform
    # whitening
    bn, cn, wn, hn = base.size()
    bv = base.view(bn, cn, wn * hn)  # vectorize feature map
    mu = th.mean(bv, dim=2, keepdim=True)  # get mean
    ss = th.std(bv, dim=2, keepdim=True)
    b = (bv - mu) / th.clamp(ss, min=1e-6)  # normalize

    # color transfer
    bv2 = base2.view(bn, cn, wn * hn)  # vectorize feature map
    mu2 = th.mean(bv2, dim=2, keepdim=True)  # get mean
    ss2 = th.std(bv2, dim=2, keepdim=True)

    bvst = b * ss2 + mu2
    return bvst.view(bn, cn, wn, hn)
