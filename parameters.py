def init_transformer_params(num_heads=64,
                            hidden_size=512,
                            dropout=0.1,
                            num_layers=6):
    if 128 % num_heads != 0:
        raise ValueError("128 should be divisible by num_heads")
    params = {
        "input_size": 128,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "num_layers": num_layers,
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
                    conv_3_stride=1):
    params = {
        "encoder": {
            "conv1": {
                "in_channels": 4,
                "out_channels": conv1_out_channels,
                "kernel_size": conv1_kernel_size,
                "stride": conv_1_stride
            },
            "conv2": {
                "in_channels": conv1_out_channels,
                "out_channels": conv2_out_channels,
                "kernel_size": conv2_kernel_size,
                "stride": conv_2_stride
            },
            "conv3": {
                "in_channels": conv2_out_channels,
                "out_channels": conv3_out_channels,
                "kernel_size": conv3_kernel_size,
                "stride": conv_3_stride
            },
        },
        "decoder": {
            "deconv1": {
                "in_channels": conv3_out_channels,
                "out_channels": conv2_out_channels,
                "kernel_size": conv3_kernel_size,
                "stride": conv_3_stride
            },
            "deconv2": {
                "in_channels": conv2_out_channels,
                "out_channels": conv1_out_channels,
                "kernel_size": conv2_kernel_size,
                "stride": conv_2_stride
            },
            "deconv3": {
                "in_channels": conv1_out_channels,
                "out_channels": 4,
                "kernel_size": conv1_kernel_size,
                "stride": conv_1_stride
            },
        },
    }
    return params


def init_fc_cnn_params(conv_1_in_channels=4,
                       conv_1_out_channels=6,
                       conv_1_kernel_size=3,
                       conv_1_stride=1,
                       conv_2_out_channels=8,
                       conv_2_kernel_size=3,
                       conv_2_stride=1,
                       flatten_size=(8, 28),
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
                "stride": conv_1_stride
            },
            "conv2": {
                "in_channels": conv_1_out_channels,
                "out_channels": conv_2_out_channels,
                "kernel_size": conv_2_kernel_size,
                "stride": conv_2_stride
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
                "stride": conv_2_stride
            },
            "conv2": {
                "in_channels": conv_1_out_channels,
                "out_channels": conv_1_in_channels,
                "kernel_size": conv_1_kernel_size,
                "stride": conv_1_stride
            }
        }
    }

    return params
