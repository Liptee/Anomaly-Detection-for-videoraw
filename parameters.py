def init_transformer_params(num_heads=36,
                            hidden_size=512,
                            dropout=0.1,
                            num_layers=4):
    if 72 % num_heads != 0:
        raise ValueError("72 should be divisible by num_heads")
    params = {
        "input_size": 72,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "num_layers": num_layers,
    }
    return params

def init_rnn_params(input_size=72,
                    hidden_size=512):
    params = {
        "input_size": input_size,
        "hidden_size": hidden_size,
    }
    return params


def init_cnn_params(conv1_out_channels=3,
                    conv2_out_channels=2,
                    conv3_out_channels=1,
                    conv1_kernel_size=3,
                    conv2_kernel_size=3,
                    conv3_kernel_size=3,
                    conv_1_stride=1,
                    conv_2_stride=1,
                    conv_3_stride=1,
                    conv_1_padding=0,
                    conv_2_padding=0,
                    conv_3_padding=0):
    params = {
        "encoder": {
            "conv1": {
                "in_channels": 3,
                "out_channels": conv1_out_channels,
                "kernel_size": conv1_kernel_size,
                "stride": conv_1_stride,
                "padding": conv_1_padding
            },
            "conv2": {
                "in_channels": conv1_out_channels,
                "out_channels": conv2_out_channels,
                "kernel_size": conv2_kernel_size,
                "stride": conv_2_stride,
                "padding": conv_2_padding
            },
            "conv3": {
                "in_channels": conv2_out_channels,
                "out_channels": conv3_out_channels,
                "kernel_size": conv3_kernel_size,
                "stride": conv_3_stride,
                "padding": conv_3_padding
            },
        },
        "decoder": {
            "deconv1": {
                "in_channels": conv3_out_channels,
                "out_channels": conv2_out_channels,
                "kernel_size": conv3_kernel_size,
                "stride": conv_3_stride,
                "padding": conv_3_padding
            },
            "deconv2": {
                "in_channels": conv2_out_channels,
                "out_channels": conv1_out_channels,
                "kernel_size": conv2_kernel_size,
                "stride": conv_2_stride,
                "padding": conv_2_padding
            },
            "deconv3": {
                "in_channels": conv1_out_channels,
                "out_channels": 3,
                "kernel_size": conv1_kernel_size,
                "stride": conv_1_stride,
                "padding": conv_1_padding
            },
        },
    }
    return params


def init_fc_cnn_params(conv_1_in_channels=3,
                       conv_1_out_channels=18,
                       conv_1_kernel_size=3,
                       conv_1_stride=1,
                       conv_1_padding=0,
                       conv_2_out_channels=36,
                       conv_2_kernel_size=3,
                       conv_2_stride=1,
                       conv_2_padding=0,
                       flatten_size=(8, 20), # (sequnce_length: 12, num_au: 24)
                       fc1_out_features=128,
                       fc2_out_features=64
                       ):
    FLATTEN_SIZE = conv_2_out_channels * flatten_size[0] * flatten_size[1]
    params = {
        "encoder": {
            "conv1": {
                "in_channels": conv_1_in_channels,
                "out_channels": conv_1_out_channels,
                "kernel_size": conv_1_kernel_size,
                "stride": conv_1_stride,
                "padding": conv_1_padding
            },
            "conv2": {
                "in_channels": conv_1_out_channels,
                "out_channels": conv_2_out_channels,
                "kernel_size": conv_2_kernel_size,
                "stride": conv_2_stride,
                "padding": conv_2_padding
            },
            "fc1": {
                "in_features": FLATTEN_SIZE,
                "out_features": fc1_out_features
            },
            "fc2": {
                "in_features": fc1_out_features,
                "out_features": fc2_out_features
            }
        },
        "decoder": {
            "fc1": {
                "in_features": fc2_out_features,
                "out_features": fc1_out_features
            },
            "fc2": {
                "in_features": fc1_out_features,
                "out_features": FLATTEN_SIZE
            },
            "unflatten": {
                "unflattened_size": (conv_2_out_channels, flatten_size[0], flatten_size[1])
            },
            "conv1": {
                "in_channels": conv_2_out_channels,
                "out_channels": conv_1_out_channels,
                "kernel_size": conv_2_kernel_size,
                "stride": conv_2_stride,
                "padding": conv_2_padding
            },
            "conv2": {
                "in_channels": conv_1_out_channels,
                "out_channels": conv_1_in_channels,
                "kernel_size": conv_1_kernel_size,
                "stride": conv_1_stride,
                "padding": conv_1_padding
            }
        }
    }

    return params

def init_lstm_params(input_size=72,
                     hidden_size=512):
    params = {
        "input_size": input_size,
        "hidden_size": hidden_size
    }
    return params