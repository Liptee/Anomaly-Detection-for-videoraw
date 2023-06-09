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
                               stride=params["conv1"]["stride"])
        self.conv2 = nn.Conv2d(in_channels=params["conv2"]["in_channels"],
                               out_channels=params["conv2"]["out_channels"],
                               kernel_size=params["conv2"]["kernel_size"],
                               stride=params["conv2"]["stride"])
        self.conv3 = nn.Conv2d(in_channels=params["conv3"]["in_channels"],
                               out_channels=params["conv3"]["out_channels"],
                               kernel_size=params["conv3"]["kernel_size"],
                               stride=params["conv3"]["stride"])

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # Should I use ReLU here?
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        return x


class CNN_Decoder(nn.Module):
    def __init__(self, params: dict):
        super(CNN_Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=params["deconv1"]["in_channels"],
                                          out_channels=params["deconv1"]["out_channels"],
                                          kernel_size=params["deconv1"]["kernel_size"],
                                          stride=params["deconv1"]["stride"])
        self.deconv2 = nn.ConvTranspose2d(in_channels=params["deconv2"]["in_channels"],
                                          out_channels=params["deconv2"]["out_channels"],
                                          kernel_size=params["deconv2"]["kernel_size"],
                                          stride=params["deconv2"]["stride"])
        self.deconv3 = nn.ConvTranspose2d(in_channels=params["deconv3"]["in_channels"],
                                          out_channels=params["deconv3"]["out_channels"],
                                          kernel_size=params["deconv3"]["kernel_size"],
                                          stride=params["deconv3"]["stride"])

    def forward(self, x):
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))

        return x
