import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RNN(nn.Module):
    def __init__(self, params):
        super(RNN, self).__init__()
        self.encoder = nn.RNN(params["input_size"], params["hidden_size"], batch_first=True)
        self.decoder = nn.RNN(params["hidden_size"], params["input_size"], batch_first=True)

    def forward(self, x):
        encoded_output, hidden_state = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()

        self.encoder = nn.LSTM(params["input_size"], params["hidden_size"], batch_first=True)
        self.decoder = nn.LSTM(params["hidden_size"], params["input_size"], batch_first=True)

    def forward(self, x):
        encoded_output, hidden_state = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        return decoded_output


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]

        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        encoder_layer = TransformerEncoderLayer(self.hidden_size, nhead=4)
        self.encoder = TransformerEncoder(encoder_layer, self.num_layers)
        decoder_layer = TransformerEncoderLayer(self.hidden_size, nhead=4)
        self.decoder = TransformerEncoder(decoder_layer, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        output = self.fc(decoded)

        return output


