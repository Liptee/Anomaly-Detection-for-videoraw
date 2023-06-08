import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, params: dict):
        if params["input_size"] % params["num_heads"] != 0:
            raise ValueError("input_size should be divisible by num_heads")
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=params["input_size"],
                nhead=params["num_heads"],
                dim_feedforward=params["hidden_size"],
                dropout=params["dropout"]),
            num_layers=params["num_layers"])

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=params["input_size"],
                nhead=params["num_heads"],
                dim_feedforward=params["hidden_size"],
                dropout=params["dropout"]),
            num_layers=params["num_layers"])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return decoded


class CNN(nn.Module):
    def __init__(self, params: dict):
        super(CNN, self).__init__()
        self.encoder = CNN_Encoder(params["encoder"])
        self.decoder = CNN_Decoder(params["decoder"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class CNN_Encoder(nn.Module):
    def __init__(self, params: dict):
        super(CNN_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=params["conv1"]["in_channels"],
                               out_channels=params["conv1"]["out_channels"],
                               kernel_size=params["conv1"]["kernel_size"],
                               stride=params["conv1"]["stride"],
                               padding=params["conv1"]["padding"])
        self.conv2 = nn.Conv2d(in_channels=params["conv2"]["in_channels"],
                               out_channels=params["conv2"]["out_channels"],
                               kernel_size=params["conv2"]["kernel_size"],
                               stride=params["conv2"]["stride"],
                               padding=params["conv2"]["padding"])
        self.conv3 = nn.Conv2d(in_channels=params["conv3"]["in_channels"],
                               out_channels=params["conv3"]["out_channels"],
                               kernel_size=params["conv3"]["kernel_size"],
                               stride=params["conv3"]["stride"],
                               padding=params["conv3"]["padding"])

    def forward(self, x):
        x = self.conv1(x)  # Should I use ReLU here?
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class CNN_Decoder(nn.Module):
    def __init__(self, params: dict):
        super(CNN_Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=params["deconv1"]["in_channels"],
                                          out_channels=params["deconv1"]["out_channels"],
                                          kernel_size=params["deconv1"]["kernel_size"],
                                          stride=params["deconv1"]["stride"],
                                          padding=params["deconv1"]["padding"])
        self.deconv2 = nn.ConvTranspose2d(in_channels=params["deconv2"]["in_channels"],
                                          out_channels=params["deconv2"]["out_channels"],
                                          kernel_size=params["deconv2"]["kernel_size"],
                                          stride=params["deconv2"]["stride"],
                                          padding=params["deconv2"]["padding"])
        self.deconv3 = nn.ConvTranspose2d(in_channels=params["deconv3"]["in_channels"],
                                          out_channels=params["deconv3"]["out_channels"],
                                          kernel_size=params["deconv3"]["kernel_size"],
                                          stride=params["deconv3"]["stride"],
                                          padding=params["deconv3"]["padding"])

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x


if __name__ == "__main__":
    params = {
        "encoder": {
            "conv1": {
                "in_channels": 1,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            "conv2": {
                "in_channels": 16,
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            "conv3": {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
        },
        "decoder": {
            "deconv1": {
                "in_channels": 64,
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            "deconv2": {
                "in_channels": 32,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            },
            "deconv3": {
                "in_channels": 16,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1
            }
        }
    }
    cnn = CNN(params)
    print(cnn)

    params = {
        "input_size": 32,
        "num_heads": 8,
        "hidden_size": 512,
        "dropout": 0.1,
        "num_layers": 6
    }
    transformer = Transformer(params)
    print(transformer)
