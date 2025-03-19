import torch


class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim=64, max_seq_length=128*88*4/64):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, int(max_seq_length), embed_dim),
            requires_grad=True
        )
        torch.nn.init.normal_(self.pos_embed, std=0.02)  # Xavier-style initialization

    def forward(self, x):
        B, seq, embed_size = x.shape
        return x + self.pos_embed[:, :seq]


class HybridTransformerBase(torch.nn.Module):
            def __init__(self, embed_dim=64, num_heads=4, num_layers=4):
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


class HybridTransformerTiny(torch.nn.Module):
            def __init__(self, embed_dim=16, num_heads=1, num_layers=1):
                super().__init__()

                # CNN Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 8, 3, padding=1),
                    torch.nn.BatchNorm2d(8),
                    torch.nn.GELU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(8, 16, 3, padding=1),
                    torch.nn.BatchNorm2d(16),
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
                    torch.nn.ConvTranspose2d(16, 16, 3, padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm2d(8),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
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
            

class HybridTransformerSmall(torch.nn.Module):
            def __init__(self, embed_dim=32, num_heads=2, num_layers=2):
                super().__init__()

                # CNN Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 16, 3, padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.GELU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(16, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
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
                    torch.nn.ConvTranspose2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
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
            

class HybridTransformerLarge(torch.nn.Module):
            def __init__(self, embed_dim=96, num_heads=8, num_layers=8):
                super().__init__()

                # CNN Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 48, 3, padding=1),
                    torch.nn.BatchNorm2d(48),
                    torch.nn.GELU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(48, 96, 3, padding=1),
                    torch.nn.BatchNorm2d(96),
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
                    torch.nn.ConvTranspose2d(96, 96, 3, padding=1),
                    torch.nn.BatchNorm2d(96),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(96, 48, 3, stride=2, padding=1, output_padding=1),
                    torch.nn.BatchNorm2d(48),
                    torch.nn.GELU(),
                    torch.nn.ConvTranspose2d(48, 1, 3, stride=2, padding=1, output_padding=1),
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