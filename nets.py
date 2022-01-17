import torch
import torch.nn as nn
from torch.nn import init
import functools
from config import *

def define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type='normal', init_gain=0.02, gpu_ids=[], load=False, name = None):
    net = None
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    return init_net(net, init_type, init_gain, gpu_ids, load, name)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type='reflect', use_bias=True):
        super(ResnetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                      kernel_size=3, stride=2,
                      padding=1, output_padding=1,
                      bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        #conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x) 
        return out

# per convertire il modello dataparallel in uno normale
#def unwrap(net):
#    from collections import OrderedDict
#    #print(net.__class__.__name__)
#    unwrapped = OrderedDict()
#    for element in net.items():
#        key = element[0][7:]
#        data = element[1]
#        unwrapped.__setitem__(key,data)
#    #print(unwrapped)
#    return unwrapped

def init_net(net, init_type, init_gain, gpu_ids, load = False, name = None):
    net.to(gpu_ids[0]) # gpu_ids = [0] (va bene pure net.to(torch.devide('cuda:0')))
    #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs non ho capito che fa esattamente? se ho capito bene ricopia il modulo in ogni device e fa tipo albero con i gradienti, cambia anche la classe di net in dataparallel
    if not load:
        init_weights(net, init_type, init_gain=init_gain)
    elif load:
        loaded = torch.load(f"{savedir}/{name}.rete")
        #loaded = unwrap(loaded) # per convertire il modello dataparallel in uno normale tanto ho solo una gpu mannaggia
        net.load_state_dict(loaded)
    return net

def init_weights(net, init_type, init_gain):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): # se m è un layer di convoluzione o lineare
            init.normal_(m.weight.data, 0.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies. # se invece m è un layer di normalizzazione
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>

def define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpu_ids=[], load = False, name = None):
    net = None
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = NLayerDiscriminator(norm_layer=norm_layer, input_nc=3, ndf=64, n_layers=3 )
    return init_net(net, init_type, init_gain, gpu_ids, load, name)

class NLayerDiscriminator(nn.Module):
    def __init__(self, norm_layer, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        use_bias = True
        kw = 4
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        return self.model(input)
