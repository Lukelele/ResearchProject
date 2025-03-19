import torch

class ConvolutionAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1),  # Output: (16, 60, 45)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 16, 3, stride=1), # Output: (32, 30, 23)
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, 3, stride=1), # Output: (64, 15, 12)
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            # torch.nn.Flatten(1),
            # torch.nn.Linear(64*114*84, 128),
            # torch.nn.ReLU(),
        )
        
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(128, 64*114*84),
            # torch.nn.ReLU(),
            # torch.nn.Unflatten(1, (64, 114, 84)),
            torch.nn.ConvTranspose2d(8, 16, 3, stride=1), # Output: (32, 30, 23)
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 64, 3, stride=1), # Output: (16, 60, 45)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, 3, stride=1), # Output: (1, 120, 90)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x