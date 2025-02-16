import sys
import torch
import TorchData
import cmd_core
import uuid

class HybridTransformer(torch.nn.Module):
    def __init__(self, embed_dim=64, num_heads=1, num_layers=2):
        super().__init__()

        # CNN Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(4, 32),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(8, 64),
            torch.nn.MaxPool2d(2)
        )

        # Transformer
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )

        # CNN Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(8, 64),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(4, 32),
            torch.nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

        self.pos_embed = torch.nn.Parameter(torch.randn(1, 690, embed_dim))

    def forward(self, x):
        # Encoder
        cnn_features = self.encoder(x)  # [B, 64, H/4, W/4]

        # Transformer processing
        B, C, H, W = cnn_features.shape
        x = cnn_features.flatten(2).permute(0, 2, 1)  # [B, Seq, C]

        x = x + self.pos_embed[:, :H * W, :]  # Positional embedding
        x = self.transformer(x)

        # Feature fusion
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + cnn_features  # Residual connection

        # Decoder pathway
        return self.decoder(x)

if __name__ == "__main__":
    if int(len(sys.argv)) == 4 or int(len(sys.argv)) == 5:
        NUM_DATA, BATCH_SIZE, EPOCH, MODEL_PATH = cmd_core.start(sys.argv, "HybridTransformer")
        train_dataset = TorchData.TORCHDataset(num_data=NUM_DATA)
        train_dataloader = train_dataset.dataloader(batch_size=BATCH_SIZE, shuffle=True)
        cmd_core.main(train_dataloader, HybridTransformer, EPOCH, MODEL_PATH)
    else:
        print("Usage: python {} <NUM_DATA> <BATCH_SIZE> <EPOCH> <MODEL_PATH>".format(sys.argv[0]))
        sys.exit(1)
