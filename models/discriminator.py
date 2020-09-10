import numpy as np
from torch import nn

class Discriminator(nn.Module):

    def __init__(self,
                 input_nc=6,
                 ndf=32,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False,
                 num_D=1,
                 getIntermFeat=False,
                 use_sn_discriminator=False
        ):
        super().__init__()

        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.use_sn_discriminator = use_sn_discriminator

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc,
                                       ndf,
                                       n_layers,
                                       norm_layer,
                                       use_sigmoid,
                                       getIntermFeat,
                                       self.use_sn_discriminator)

            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, f'scale{i}_layer{j}',
                            getattr(netD, f'model{j}'))
            else:
                setattr(self, f'layer{i}', netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            shapes = [input.shape]
            for i in range(len(model)):
                result.append(model[i](result[-1]))

            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):

        num_D = self.num_D
        result = []
        shapes = []
        input_downsampled = input

        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, f'scale{num_D - 1 - i}_layer{j}') \
                         for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, f'layer{num_D - 1 - i}')
    
            res = self.singleD_forward(model, input_downsampled)
            result.append(res)

            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 getIntermFeat=False,
                 use_sn_discriminator=False
        ):
        super().__init__()
        
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.use_sn_discriminator = use_sn_discriminator
        
        kw = 4
        padw = int(np.floor((kw-1.0)/2))

        sequence = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)

        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[
            nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        if use_sigmoid:
            sequence += [[
                nn.Sigmoid()
            ]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, f'model{n}', nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, f'model{n}')
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)