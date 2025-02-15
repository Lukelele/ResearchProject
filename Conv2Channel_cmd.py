import sys
import torch
import TorchData
import cmd_core

class ConvolutionAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 64, 3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 16, 3, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, 3, stride=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            #torch.nn.Flatten()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 16, 3, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 64, 3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 2, 3, stride=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    if int(len(sys.argv)) == 4 or int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        NUM_DATA = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
        EPOCH = int(sys.argv[3])
        MODEL_PATH = "Conv2Channel"
        if int(len(sys.argv)) == 5:
            MODEL_PATH = sys.argv[4]
        train_dataset = TorchData.TORCHDataset2Channel(num_data=NUM_DATA)
        train_dataloader = train_dataset.dataloader(batch_size=BATCH_SIZE, shuffle=True)
        cmd_core.main(train_dataloader, ConvolutionAutoencoder, EPOCH, MODEL_PATH)
    else:
        print("Usage: python {} <NUM_DATA> <BATCH_SIZE> <EPOCH> <MODEL_PATH>".format(sys.argv[0]))
        sys.exit(1)
