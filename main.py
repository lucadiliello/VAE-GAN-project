from argparse import ArgumentParser
from model import VaeGanModule
import pytorch_lightning import Trainer

def main(hparams):
    model = VaeGanModule()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = VaeGanModel.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)