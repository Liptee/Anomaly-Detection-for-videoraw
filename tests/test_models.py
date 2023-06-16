from parameters import init_transformer_params, init_cnn_params, init_fc_cnn_params
from models import Transformer, CNN, FC_CNN
import unittest
from tqdm import tqdm


class TestModels(unittest.TestCase):
    def test_default_transformer_init(self):
        params = init_transformer_params()
        Transformer(params)

    def test_default_cnn_init(self):
        params = init_cnn_params()
        CNN(params)

    def test_default_fc_cnn_init(self):
        params = init_fc_cnn_params()
        FC_CNN(params)


    def test_transformer_with_different_params(self):
        for num_heads in tqdm([2, 4, 8, 16, 32, 64]):
            for hidden_size in range(100, 1000, 20):
                for dropout in [0.0, 0.1, 0.5]:
                    for num_layers in range(2, 11, 3):
                        params = init_transformer_params(num_heads=num_heads,
                                                         hidden_size=hidden_size,
                                                         dropout=dropout,
                                                         num_layers=num_layers)
                        Transformer(params)

    def test_cnn_with_different_params(self):
        for conv1_out_channels in tqdm([2, 4, 8, 16, 32, 64]):
            for conv2_out_channels in [2, 4, 8, 16, 32, 64]:
                for conv3_out_channels in [2, 4, 8, 16, 32, 64]:
                    for conv1_kernel_size in [1, 3, 5, 7]:
                        for conv2_kernel_size in [1, 3, 5, 7]:
                            for conv3_kernel_size in [1, 3, 5, 7]:
                                for conv_1_stride in [1, 2, 3]:
                                    for conv_2_stride in [1, 2, 3]:
                                        for conv_3_stride in [1, 2, 3]:
                                            params = init_cnn_params(conv1_out_channels=conv1_out_channels,
                                                                     conv2_out_channels=conv2_out_channels,
                                                                     conv3_out_channels=conv3_out_channels,
                                                                     conv1_kernel_size=conv1_kernel_size,
                                                                     conv2_kernel_size=conv2_kernel_size,
                                                                     conv3_kernel_size=conv3_kernel_size,
                                                                     conv_1_stride=conv_1_stride,
                                                                     conv_2_stride=conv_2_stride,
                                                                     conv_3_stride=conv_3_stride)
                                            CNN(params)

    def test_fc_cnn_with_different_params(self):
        for conv_1_in_channels in tqdm([4, 12, 32]):
            for conv_1_out_channels in [2, 4, 8, 16]:
                for conv_2_out_channels in [2, 4, 8, 16]:
                    for conv_1_kernel_size in [1, 3, 5, 7]:
                        for conv_2_kernel_size in [1, 3, 5, 7]:
                            for conv_1_stride in [1, 2, 3]:
                                for conv_2_stride in [1, 2, 3]:
                                    for fc1_out_features in [128, 256, 512]:
                                        for fc2_out_features in [128, 256, 512]:
                                            params = init_fc_cnn_params(conv_1_in_channels=conv_1_in_channels,
                                                                        conv_1_out_channels=conv_1_out_channels,
                                                                        conv_2_out_channels=conv_2_out_channels,
                                                                        conv_1_kernel_size=conv_1_kernel_size,
                                                                        conv_2_kernel_size=conv_2_kernel_size,
                                                                        conv_1_stride=conv_1_stride,
                                                                        conv_2_stride=conv_2_stride,
                                                                        fc1_out_features=fc1_out_features,
                                                                        fc2_out_features=fc2_out_features)
                                            FC_CNN(params)
