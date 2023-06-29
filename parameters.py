def init_transformer_params(input_size=72,
                            hidden_size=512,
                            num_layers=4):
    params = {
        "input_size": input_size,
        "hidden_size": hidden_size,
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


def init_lstm_params(input_size=72,
                     hidden_size=512):
    params = {
        "input_size": input_size,
        "hidden_size": hidden_size
    }
    return params