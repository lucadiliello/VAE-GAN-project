import pytorch_lightning as pl
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from losses import losses
from collections import OrderedDict
import torchvision


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class Encoder(nn.Module):

    def __init__(self, ngf=32, z_dim=512):
        super(Encoder, self).__init__()

        self.f_dim = ngf
        self.z_dim = z_dim
        self.input_dim = 3
    
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.input_dim, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.f_dim * 2, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.ELU(),
        )
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * 128, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.ReLU(True))

        self.fc_mu = nn.Sequential(
            nn.Linear(1024, self.z_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(1024, self.z_dim)
        )

    def forward(self, img):
        e0 = self.conv0(img)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e2 = e2.view(e2.shape[0], -1)
        e3 = self.fc(e2)
        fc_mu = self.fc_mu(e3)
        logvar = self.fc_logvar(e3)

        return fc_mu, logvar


class Decoder(nn.Module):

    def __init__(self, ngf=32, z_dim=512):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.f_dim = ngf
        self.lin0 = nn.Sequential(
            nn.Linear(self.z_dim, 128 * 8 * 8),
            nn.ELU()
        )
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 4, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 2, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim, 3,
                      kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):

        dec1 = self.lin0(z)

        dec1 = dec1.view(dec1.shape[0], self.f_dim * 4, 8 , 8)
    
        dec2 = self.conv0(dec1)
        dec3 = self.conv1(dec2)
        instance = self.conv2(dec3)

        return instance


class Discriminator(nn.Module):

    def __init__(self, input_nc=6, ndf=32, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=1, getIntermFeat=False,
                 use_sn_discriminator=False):
        super(Discriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.use_sn_discriminator = use_sn_discriminator
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                                       use_sigmoid, getIntermFeat,
                                       self.use_sn_discriminator)
            if getIntermFeat:

                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

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
                model = [getattr(self,
                                 'scale' + str(num_D - 1 - i) + '_layer' + str(
                                     j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            res = self.singleD_forward(model, input_downsampled)
            result.append(res)
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False,
                 use_sn_discriminator=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.use_sn_discriminator = use_sn_discriminator
        kw = 4
        padw = int(np.floor((kw-1.0)/2))
        sequence = [
            [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
             nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        print("Done!")
    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class VaeGanModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.ngf = hparams.ngf
        self.z_dim = hparams.z_dim
        self.hparams = hparams
        self.encoder = Encoder(ngf=self.ngf, z_dim=self.z_dim)
        self.encoder.apply(weights_init)
        self.decoder = Decoder(ngf=self.ngf, z_dim=self.z_dim)
        self.decoder.apply(weights_init)
        self.discriminator = Discriminator()
        self.discriminator.apply(weights_init)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionGAN = losses.GANLoss(gan_mode="lsgan")
        self.last_imgs = None

    def reparameterize(self, mu, logvar, mode):
        if mode == 'train':
            std = torch.exp(0.5 * logvar)
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + eps * std
        else:
            return mu

    def discriminate(self, fake_image, real_image):
        input_concat_fake = \
            torch.cat((fake_image.detach(), real_image), dim=1) # non sono sicuro che .detach() sia necessario in lightning
        input_concat_real = \
            torch.cat((real_image, real_image),
                      dim=1)
        return self.discriminator.forward(input_concat_fake), \
               self.discriminator.forward(input_concat_real)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        self.last_imgs = x
        if optimizer_idx == 0:
            # encode
            mu, log_var = self.encoder(x)
            z_repar = self.reparameterize(mu, log_var, mode='train')
            # decode
            fake_image = self.decoder(z_repar)
            # reconstruction
            reconstruction_loss = self.criterionFeat(fake_image, x)
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
                dim=0)
            input_concat_fake = \
                torch.cat((fake_image, x), dim=1)
            pred_fake = self.discriminator.forward(input_concat_fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            g_loss = reconstruction_loss + kld_loss + loss_G_GAN
            result = pl.TrainResult(g_loss)
            result.log("rec_loss", reconstruction_loss, prog_bar=True)
            result.log("loss_G_GAN", loss_G_GAN, prog_bar=True)
            result.log("kld_loss", kld_loss, prog_bar=True)

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # encode
            mu, log_var = self.encoder(x)
            z_repar = self.reparameterize(mu, log_var, mode="train")
            # decode
            fake_image = self.decoder(z_repar)
            # how well can it label as real?
            pred_fake, pred_real = self.discriminate(fake_image, x)
            loss_D_fake = self.criterionGAN.forward(pred_fake, False)

            # Real Loss

            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            result = pl.TrainResult(loss_D)
            result.log("loss_D_real", loss_D_real, prog_bar=True)
            result.log("loss_D_fake", loss_D_fake, prog_bar=True)
        
        return result

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, log_var = self.encoder(x)
        z_repar = self.reparameterize(mu, log_var, mode='train')
        recons = self.decoder(z_repar)
        reconstruction_loss = nn.functional.mse_loss(recons, x)

        result = pl.EvalResult(checkpoint_on=reconstruction_loss)
        return result

    testing_step = validation_step

    def on_epoch_end(self):
        z_appr = torch.FloatTensor(16, self.hparams.z_dim).normal_(0, 1)
        # match gpu device (or keep as cpu)
        # log sampled images
        if self.on_gpu:
            z_appr = z_appr.cuda(self.last_imgs.device.index)
        sample_imgs = self.decoder(z_appr)
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, range=(-1,1))
        torchvision.utils.save_image(sample_imgs, f"generated_images_{self.current_epoch}.png", normalize=True, range=(-1,1))
        self.logger.experiment.add_image(f'generated_images', grid,
                                         self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        params_vae = list(self.encoder.parameters()) + \
                          list(self.decoder.parameters())
        opt_vae = torch.optim.Adam(params_vae, lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_vae, opt_d], []

    @staticmethod
    def add_argparse_args(parser):

        parser.add_argument('--ngf', type=int, default=128)
        parser.add_argument('--z_dim', type=int, default=128)

        parser.add_argument('--b1', type=float, default=0.0,
                             help='momentum term of adam')
        parser.add_argument('--b2', type=float, default=0.9,
                             help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')

        return parser