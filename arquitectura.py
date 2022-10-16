
import torch.nn as nn

class AutoEncoderV2(nn.Module):
    def __init__(
        self,
        # kernel_conv
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            # # Bloque 2
            nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            # # Bloque 2
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7)
        )

        

        self.decoder = nn.Sequential(
            # BLoque 1
            nn.ConvTranspose2d(64, 32, 5,stride=1),
            nn.ReLU(),
            nn.Upsample(size=(7,7), mode='nearest'),

            # BLoque 2
            nn.ConvTranspose2d(32, 16, 5,stride=1),
            nn.ReLU(),
            nn.Upsample(size=(21,21), mode='nearest'),

            # BLoque 3
            nn.ConvTranspose2d(16, 3, 5,stride=1),
            nn.ReLU(),
            nn.Upsample(size=(63,63), mode='nearest'),
        )
        
        self.net = nn.Sequential(
            self.encoder,
            # self.code,
            self.decoder,
        )

    def forward(self, x):
        return self.net(x)



class AutoEncoderV3(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # Bloque 1
            # [3x21x21]
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            #[16, 21, 21]
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            #[16, 21, 21]
            nn.MaxPool2d(kernel_size=2),
            #[16, 10, 10]

            # Bloque 2
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            #[32, 10, 10]
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            #[32, 10, 10]
            nn.MaxPool2d(kernel_size=2),
            # [32, 5, 5]

            # Bloque 3
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            # [64, 5, 5]
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            # [64, 5, 5]
            nn.MaxPool2d(kernel_size=2),
            # [64, 2, 2]

            # Bloque 4
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            # [128, 2, 2]
            nn.MaxPool2d(kernel_size=2),
            # [128, 1, 1]   
        )
        self.decoder = nn.Sequential(
            # BLoque 1
            # [128, 1, 1]
            nn.ConvTranspose2d(128, 64, 2,stride=1),
            nn.ReLU(),
            # [64, 2, 2]

            # BLoque 2
            nn.ConvTranspose2d(64, 64, 3,stride=2),
            nn.ReLU(),
            # [64, 5, 5]
            nn.ConvTranspose2d(64, 32, 1,stride=1),
            nn.ReLU(),
            # [32, 5, 5]

            # Bloque 3
            nn.ConvTranspose2d(32, 32, 2,stride=2),
            nn.ReLU(),
            # [32, 10, 10]
            nn.ConvTranspose2d(32, 32, 1,stride=1),
            nn.ReLU(),
            # [32, 10, 10]

            # BLoque 4
            nn.ConvTranspose2d(32, 16, 3,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 1,stride=1),
            nn.ReLU(),
            # Bloque 5
            nn.ConvTranspose2d(16, 3, 1,stride=1),
            nn.ReLU(),
        )
        
        self.net = nn.Sequential(
            self.encoder,
            self.decoder,
        )
    def forward(self, x):
        return self.net(x)