from torch import nn

class Decoder(nn.Module):

    def __init__(self, ngf=32, z_dim=512):
        super().__init__()

        self.z_dim = z_dim
        self.f_dim = ngf

        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.f_dim * 16, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(self.f_dim * 16, affine=False),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim * 16, self.f_dim * 32,
                      kernel_size=3, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim * 8, self.f_dim * 16,
                      kernel_size=3, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.ELU(),
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim * 4, self.f_dim * 8,
                      kernel_size=3, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim * 2, self.f_dim * 4,
                      kernel_size=3, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ELU()
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim, self.f_dim * 4,
                               kernel_size=3, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.f_dim, 3,
                      kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, z):

        dec0 = self.conv0(z.unsqueeze(2).unsqueeze(2))
        dec1 = self.conv1(dec0)
        dec2 = self.conv2(dec1)
        dec3 = self.conv3(dec2)
        dec4 = self.conv4(dec3)

        instance = self.conv5(dec4)

        return instance