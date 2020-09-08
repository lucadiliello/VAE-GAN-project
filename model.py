import pytorch_lightning as pl
from torch import nn

class VaeGanModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28)
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        # decode
        recons = self.decoder(z)

        # reconstruction
        reconstruction_loss = nn.functional.mse_loss(recons, x)
        return pl.TrainResult(reconstruction_loss)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recons = self.decoder(z)
        reconstruction_loss = nn.functional.mse_loss(recons, x)

        result = pl.EvalResult(checkpoint_on=reconstruction_loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    @staticmethod
    def add_argparse_args(parser):
        # add some arguments to the cmd that
        # are need by this model
        # parser.add_argument("--number_layers", type=int, ...)
        return parser