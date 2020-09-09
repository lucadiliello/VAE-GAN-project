from torch import nn

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
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.f_dim * 4, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.f_dim * 16,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(self.f_dim * 16, self.z_dim, kernel_size=4, stride=1, padding=0)
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(self.f_dim * 16, self.z_dim, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, img):

        e0 = self.conv0(img)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        fc_mu = self.fc_mu(e4).squeeze(2).squeeze(2)
        logvar = self.fc_logvar(e4).squeeze(2).squeeze(2)

        return fc_mu, logvar