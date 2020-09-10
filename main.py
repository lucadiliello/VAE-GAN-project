import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from models import VaeGanModule
from dataloader import CelebaDataModule

def main(hparams):

    # Prepare folders
    if not os.path.isdir(hparams.generated_images_folder):
        os.mkdir(hparams.generated_images_folder)

    data = CelebaDataModule(hparams)
    model = VaeGanModule(hparams)
    trainer = Trainer.from_argparse_args(hparams)

    # train
    trainer.fit(model, data)

if __name__ == '__main__':

    parser = ArgumentParser()
                        
    parser = Trainer.add_argparse_args(parser)
    parser = VaeGanModule.add_argparse_args(parser)
    parser = CelebaDataModule.add_dataloader_args(parser)
    args = parser.parse_args()

    main(args)