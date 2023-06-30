import sys
sys.path.append('..')
from parameters import init_transformer_params, init_rnn_params, init_lstm_params
from models import Transformer, RNN, LSTM
import unittest
from tqdm import tqdm
import torch


class TestModels(unittest.TestCase):
    def test_default_transformer_init(self):
        params = init_transformer_params()
        model = Transformer(params)
        x = torch.randn(16, 24, params["input_size"])
        model(x)

    def test_default_rnn_init(self):
        params = init_rnn_params()
        model = RNN(params)
        x = torch.randn(16, 24, params["input_size"])
        model(x)

    def test_default_lstm_init(self):
        params = init_lstm_params()
        LSTM(params)
        model = LSTM(params)
        x = torch.randn(16, 24, params["input_size"])
        model(x)

    def test_transformer_init_with_params(self):
        for input_size in tqdm(range(2, 200, 13)):
            for hidden_size in range(100, 1000, 12):
                for num_layers in [2, 4, 8]:
                    params = init_transformer_params(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers)
                    model = Transformer(params)
                    x = torch.randn(16, 24, input_size)
                    model(x)

    def test_rnn_init_with_params(self):
        for input_size in tqdm(range(2, 200, 13)):
            for hidden_size in range(25, 1000, 13):
                    params = init_rnn_params(input_size=input_size,
                                             hidden_size=hidden_size)
                    model = RNN(params)
                    x = torch.randn(16, 24, input_size)
                    model(x)

    def test_lstm_init_with_params(self):
        for input_size in tqdm(range(2, 200, 13)):
            for hidden_size in range(5, 1000, 13):
                    params = init_lstm_params(input_size=input_size,
                                              hidden_size=hidden_size)
                    model = LSTM(params)
                    x = torch.randn(16, 24, input_size)
                    model(x)