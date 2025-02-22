import os
import torch
from tqdm import tqdm
import uuid
import sys
from dotenv import load_dotenv
import time
import TorchData

device = "cuda" if torch.cuda.is_available() else "mps"

def main(model):
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    scalar = torch.amp.GradScaler()

    start_time = time.time()
    for _ in tqdm(range(EPOCH)):
        for x, y in train_dataloader:
            optimiser.zero_grad()

            with torch.amp.autocast(device):
                outputs = model(x.to(device))
                loss = loss_function(outputs, y.to(device))

            scalar.scale(loss).backward()
            scalar.step(optimiser)
            scalar.update()

    time_taken = time.time() - start_time
    torch.save(model.state_dict(), f"{output_path}.pth")
    print("Model saved to", f"{output_path}.pth")
    write_meta(time_taken)

def write_meta(time_taken):
    with open(f"{output_path}.txt", "w") as f:
        f.write("X={}\n".format(X))
        f.write("Y={}\n".format(Y))
        f.write("T_OFFSET={}\n".format(T_OFFSET))
        f.write("NUM_DATA={}\n".format(NUM_DATA))
        f.write("SIGNAL_COUNT_MIN={}\n".format(SIGNAL_COUNT_MIN))
        f.write("SIGNAL_COUNT_MAX={}\n".format(SIGNAL_COUNT_MAX))
        f.write("SIGNAL_SELECT_MODE=\"{}\"\n".format(SIGNAL_SELECT_MODE))
        f.write("NOISE_DENSITY={}\n".format(NOISE_DENSITY))
        f.write("NOISE_MAGNITUDE={}\n".format(NOISE_MAGNITUDE))
        f.write("BLUR_LEVEL={}\n".format(BLUR_LEVEL))
        f.write("DISPERSION_LEVEL={}\n".format(DISPERSION_LEVEL))
        f.write("MODE=\"{}\"\n".format(MODE))
        f.write("\n")
        f.write("BATCH_SIZE={}\n".format(BATCH_SIZE))
        f.write("EPOCH={}\n".format(EPOCH))
        f.write("MULTI_GPU={}\n".format(MULTI_GPU))
        f.write("\n")
        f.write("model_name=\"{}\"\n".format(model_name))
        f.write("output_path=\"{}\"\n".format(output_path))
        f.write("time_taken={}\n".format(time_taken))

def print_meta():
    print("X={}".format(X))
    print("Y={}".format(Y))
    print("T_OFFSET={}".format(T_OFFSET))
    print("NUM_DATA={}".format(NUM_DATA))
    print("SIGNAL_COUNT_MIN={}".format(SIGNAL_COUNT_MIN))
    print("SIGNAL_COUNT_MAX={}".format(SIGNAL_COUNT_MAX))
    print("SIGNAL_SELECT_MODE=\"{}\"".format(SIGNAL_SELECT_MODE))
    print("NOISE_DENSITY={}".format(NOISE_DENSITY))
    print("NOISE_MAGNITUDE={}".format(NOISE_MAGNITUDE))
    print("BLUR_LEVEL={}".format(BLUR_LEVEL))
    print("DISPERSION_LEVEL={}".format(DISPERSION_LEVEL))
    print("MODE=\"{}\"".format(MODE))
    print()
    print("BATCH_SIZE={}".format(BATCH_SIZE))
    print("EPOCH={}".format(EPOCH))
    print("MULTI_GPU={}".format(MULTI_GPU))
    print()
    print("model_name={}".format(model_name))
    print("output_path={}".format(output_path))


if __name__ == "__main__":
    if int(len(sys.argv)) == 3:
        load_dotenv(str(sys.argv[1]))
        model_name = str(sys.argv[2])
        X = int(os.getenv("X"))
        Y = int(os.getenv("Y"))
        T_OFFSET = int(os.getenv("T_OFFSET"))
        NUM_DATA = int(os.getenv("NUM_DATA"))
        SIGNAL_COUNT_MIN = int(os.getenv("SIGNAL_COUNT_MIN"))
        SIGNAL_COUNT_MAX = int(os.getenv("SIGNAL_COUNT_MAX"))
        SIGNAL_SELECT_MODE = os.getenv("SIGNAL_SELECT_MODE")
        NOISE_DENSITY = float(os.getenv("NOISE_DENSITY"))
        NOISE_MAGNITUDE = float(os.getenv("NOISE_MAGNITUDE"))
        BLUR_LEVEL = int(os.getenv("BLUR_LEVEL"))
        DISPERSION_LEVEL = float(os.getenv("DISPERSION_LEVEL"))
        MODE = os.getenv("MODE")

        BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
        EPOCH = int(os.getenv("EPOCH"))
        MULTI_GPU = bool(os.getenv("MULTI_GPU"))

        train_dataset = TorchData.TORCHData(x=X, y=Y, num_data=NUM_DATA,
                                            signal_count=(SIGNAL_COUNT_MIN, SIGNAL_COUNT_MAX),
                                            signal_select_mode=SIGNAL_SELECT_MODE,
                                            noise_density=NOISE_DENSITY, t_offset=T_OFFSET,
                                            blur_level=BLUR_LEVEL, dispersion_level=DISPERSION_LEVEL, mode=MODE)
        train_dataloader = train_dataset.dataloader(batch_size=BATCH_SIZE, shuffle=True)
        output_path = f"{model_name}_{uuid.uuid4()}"
        print_meta()


        class LearnablePositionalEncoding(torch.nn.Module):
            def __init__(self, embed_dim=64, max_seq_length=X * Y * 4 / 64):
                super().__init__()
                self.pos_embed = torch.nn.Parameter(
                    torch.zeros(1, int(max_seq_length), embed_dim),
                    requires_grad=True
                )
                torch.nn.init.normal_(self.pos_embed, std=0.02)  # Xavier-style initialization

            def forward(self, x):
                B, seq, embed_size = x.shape
                return x + self.pos_embed[:, :seq]

        class HybridTransformer(torch.nn.Module):
            def __init__(self, embed_dim=64, num_heads=8, num_layers=4):
                super().__init__()

                # CNN Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.GELU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.GELU(),
                    torch.nn.MaxPool2d(2)
                )

                # Transformer
                self.transformer = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=4 * embed_dim,
                        batch_first=True,
                        norm_first=True,  # Better stability
                        activation="gelu",
                        dropout=0.1
                    ),
                    num_layers=num_layers
                )

                # CNN Decoder
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                )

                # self.pos_embed = torch.nn.Parameter(torch.randn(1, 690, embed_dim))
                self.learnable_position_embed = LearnablePositionalEncoding(embed_dim)
                # self.fixed_pos_embed = PositionalEncoding(embed_dim)

            def forward(self, x):
                # Encoder
                cnn_features = self.encoder(x)  # [B, 64, H/4, W/4]

                # Transformer processing
                B, C, H, W = cnn_features.shape
                x = cnn_features.flatten(2).permute(0, 2, 1)

                # x = x + self.pos_embed[:, :H*W, :]
                # x = self.fixed_pos_embed(x)
                x = self.learnable_position_embed(x)
                x = self.transformer(x)

                # Feature fusion
                x = x.permute(0, 2, 1).view(B, C, H, W)
                x = x + cnn_features  # Residual connection

                # Decoder pathway
                return self.decoder(x)

        model = None
        if model_name == "HybridTransformer":
            model = HybridTransformer()
        main(model)
    else:
        print("Usage: python {} <ENV_FILE_NAME> <MODEL_NAME>".format(sys.argv[0]))
        sys.exit(1)