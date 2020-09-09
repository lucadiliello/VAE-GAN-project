from pytorch_lightning import LightningDataModule
from torchvision import transforms
import torchvision
import torch
from torch.utils.data import DataLoader

class CelebaDataModule(LightningDataModule):

    name = 'CelebA'

    def __init__(
        self,
        hparams,
        *args,
        **kwargs,
    ):
        """
        Specs:
            - Each image is (3 x 218 x 178)
        Transforms::
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])
        Example::
            from dataloader import CelebADataModule
            dataset = CelebADataModule('data/')
            model = StupidModel()
            Trainer().fit(model, dataset)
        """
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.transforms = self._default_transforms()

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = torchvision.datasets.ImageFolder(self.hparams.data_dir, transform=self.transforms)

        self.train, self.valid, self.test = \
            torch.utils.data.random_split(dataset, 
                                          [len(dataset) - self.hparams.val_split - self.hparams.test_split, 
                                           self.hparams.val_split,
                                           self.hparams.test_split])

        """
        dataset = torchvision.datasets.CelebA(self.hparams.data_dir,
                                              split='all',
                                              target_type=None,
                                              transform=transforms,
                                              target_transform=None,
                                              download=False)
        """

    def train_dataloader(self):
        """
        CelebA train set
        """
        loader = DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            #transforms=transforms,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        CelebA valid set
        """
        loader = DataLoader(
            self.valid,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        """
        CelebA test set
        """
        loader = DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        """
        Apply recommended transformations for CelebA
        """
        res = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return res

    @staticmethod
    def add_dataloader_args(parser):
        parser.add_argument("--batch_size", type=int, default=32)

        parser.add_argument("--dims", type=tuple, nargs="+", default=[1, 28, 28])
        parser.add_argument("--data_dir", type=str, default="./data")

        parser.add_argument("--val_split", type=int, default=10000)
        parser.add_argument("--test_split", type=int, default=20000)

        parser.add_argument("--num_workers", type=int, default=16)

        parser.add_argument("--normalize", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        return parser