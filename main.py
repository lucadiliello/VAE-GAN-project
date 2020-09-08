from argparse import ArgumentParser
from model import VaeGanModule
from dataloader import CelebaDataModule
from pytorch_lightning import Trainer

def main(hparams):
    data = CelebaDataModule(hparams)
    model = VaeGanModule(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model, data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--b1', type=float, default=0.0,
                             help='momentum term of adam')
    parser.add_argument('--b2', type=float, default=0.9,
                             help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser = Trainer.add_argparse_args(parser)
    parser = VaeGanModule.add_argparse_args(parser)
    parser = CelebaDataModule.add_dataloader_args(parser)
    args = parser.parse_args()

    main(args)