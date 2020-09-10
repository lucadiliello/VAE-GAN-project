import os
import torch
import torchvision
from torch import nn
from torchsummary import summary

import pytorch_lightning as pl

from losses.losses import GANLoss, Perceptual_Loss
from models.decoder import Decoder
from models.encoder import Encoder
from models.discriminator import Discriminator
from models.utils import weights_init, concat_generators


class VaeGanModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Encoder
        self.encoder = Encoder(ngf=self.hparams.ngf, z_dim=self.hparams.z_dim)
        self.encoder.apply(weights_init)
        summary(self.encoder.cuda(), (3,128,128))
        
        # Decoder
        self.decoder = Decoder(ngf=self.hparams.ngf, z_dim=self.hparams.z_dim)
        self.decoder.apply(weights_init)
        summary(self.decoder.cuda(), (self.z_dim,))
        
        # Discriminator
        self.discriminator = Discriminator()
        self.discriminator.apply(weights_init)

        # Losses
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionGAN = GANLoss(gan_mode="lsgan")

        if self.hparams.use_vgg:
            self.criterion_perceptual_style = Perceptual_Loss()

    @staticmethod
    def reparameterize(mu, logvar, mode='train'):
        if mode == 'train':
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def discriminate(self, fake_image, real_image):
        input_concat_fake = torch.cat((fake_image.detach(), real_image), dim=1) # non sono sicuro che .detach() sia necessario in lightning
        input_concat_real = torch.cat((real_image, real_image), dim=1)

        return (
            self.discriminator(input_concat_fake),
            self.discriminator(input_concat_real)
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        # train VAE
        if optimizer_idx == 0:

            # encode
            mu, log_var = self.encoder(x)
            z_repar = VaeGanModule.reparameterize(mu, log_var)

            # decode
            fake_image = self.decoder(z_repar)

            # reconstruction
            reconstruction_loss = self.criterionFeat(fake_image, x)
            kld_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # Discriminate
            input_concat_fake = torch.cat((fake_image, x), dim=1)
            pred_fake = self.discriminator(input_concat_fake)

            # Losses
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            if self.hparams.use_vgg:
                loss_G_perceptual = \
                    self.criterion_perceptual_style(fake_image, x)
            else:
                loss_G_perceptual = 0.0
            g_loss = (reconstruction_loss * 20) + kld_loss + loss_G_GAN + loss_G_perceptual
            
            # Results are collected in a TrainResult object
            result = pl.TrainResult(g_loss)
            result.log("rec_loss", reconstruction_loss * 10, prog_bar=True)
            result.log("loss_G_GAN", loss_G_GAN, prog_bar=True)
            result.log("kld_loss", kld_loss, prog_bar=True)
            result.log("loss_G_perceptual", loss_G_perceptual, prog_bar=True)

        # train Discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # Encode
            mu, log_var = self.encoder(x)
            z_repar = VaeGanModule.reparameterize(mu, log_var)

            # Decode
            fake_image = self.decoder(z_repar)

            # how well can it label as real?
            pred_fake, pred_real = self.discriminate(fake_image, x)
            
            # Fake loss
            d_loss_fake = self.criterionGAN(pred_fake, False)

            # Real Loss
            d_loss_real = self.criterionGAN(pred_real, True)

            # Total loss is average of prediction of fakes and reals
            loss_D = (d_loss_fake + d_loss_real) / 2

            # Results are collected in a TrainResult object
            result = pl.TrainResult(loss_D)
            result.log("loss_D_real", d_loss_real, prog_bar=True)
            result.log("loss_D_fake", d_loss_fake, prog_bar=True)

        return result

    def training_epoch_end(self, training_step_outputs):
        z_appr = torch.normal(mean=0,
                              std=1,
                              size=(16, self.hparams.z_dim),
                              device=training_step_outputs[0].minimize.device)

        # Generate images from latent vector
        sample_imgs = self.decoder(z_appr)
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, range=(-1,1))

        # where to save the image
        path = os.path.join(self.hparams.generated_images_folder,
                            f"generated_images_{self.current_epoch}.png")
        torchvision.utils.save_image(sample_imgs,
                                     path,
                                     normalize=True,
                                     range=(-1,1))

        # Log images in tensorboard
        self.logger.experiment.add_image(f'generated_images',
                                         grid,
                                         self.current_epoch)

        # Epoch level metrics
        epoch_loss = torch.mean(torch.stack([x['minimize'] for x in training_step_outputs]))
        results = pl.TrainResult()
        results.log("epoch_loss", epoch_loss, prog_bar=False)

        return results

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # Encode
        mu, log_var = self.encoder(x)
        z_repar = VaeGanModule.reparameterize(mu, log_var)

        # Decode
        recons = self.decoder(z_repar)
        reconstruction_loss = nn.functional.mse_loss(recons, x)

        # Results are collected in a EvalResult object
        result = pl.EvalResult(checkpoint_on=reconstruction_loss)
        return result

    testing_step = validation_step

    def configure_optimizers(self):
        params_vae = concat_generators(self.encoder.parameters(), self.decoder.parameters())
        opt_vae = torch.optim.Adam(params_vae, lr=self.hparams.learning_rate_vae)

        parameters_discriminator = self.discriminator.parameters()
        opt_d = torch.optim.Adam(parameters_discriminator, lr=self.hparams.learning_rate_d)

        return [opt_vae, opt_d]

    @staticmethod
    def add_argparse_args(parser):

        parser.add_argument('--generated_images_folder', required=False, default="./output", type=str)
        parser.add_argument('--ngf', type=int, default=128)
        parser.add_argument('--z_dim', type=int, default=128)
        parser.add_argument('--learning_rate_vae', default=1e-03, required=False, type=float)
        parser.add_argument('--learning_rate_d', default=1e-03, required=False, type=float)

        return parser