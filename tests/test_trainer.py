import sys
sys.path.append('..')
import unittest
from trainer import Trainer
from parameters import init_rnn_params, init_lstm_params, init_transformer_params
import pickle
from utils import make_samples
import numpy as np
import os
from utils import load_data


class TestTrainer(unittest.TestCase):
    def test_Trainer_init(self):
        params = init_rnn_params()
        trainer = Trainer(params,
                          model='rnn',
                          learning_rate=0.001,
                          batch_size=16,
                          num_epochs=2,
                          sequence_length=4)
        self.assertEqual(trainer.model_type, 'rnn')
        self.assertEqual(trainer.sequence_length, 4)
        self.assertEqual(trainer.num_epochs, 2)

        params = init_lstm_params()
        trainer = Trainer(params,
                          model='lstm',
                          learning_rate=0.001,
                          batch_size=16,
                          num_epochs=2,
                          sequence_length=4)
        self.assertEqual(trainer.model_type, 'lstm')
        self.assertEqual(trainer.sequence_length, 4)
        self.assertEqual(trainer.num_epochs, 2)

        params = init_transformer_params()
        trainer = Trainer(params,
                          model='transformer',
                          learning_rate=0.001,
                          batch_size=16,
                          num_epochs=2,
                          sequence_length=4)
        self.assertEqual(trainer.model_type, 'transformer')
        self.assertEqual(trainer.sequence_length, 4)
        self.assertEqual(trainer.num_epochs, 2)

    def test_Trainer_add_data(self):
        params = init_rnn_params(hidden_size=217)
        trainer = Trainer(params,
                          model='rnn',
                          learning_rate=0.0000001,
                          batch_size=9,
                          num_epochs=20,
                          sequence_length=16)
        trainer.add_data("")
        self.assertEqual(np.array(trainer.data).shape, (704, trainer.sequence_length, 24, 3))

        with open("cptn.pkl", "rb") as f:
            data_copy = pickle.load(f)
        data_copy = make_samples(data_copy, trainer.sequence_length)
        data_copy = np.array(data_copy)

        self.assertEqual(np.array(trainer.data).shape, data_copy.shape)
        assert np.array_equal(np.array(trainer.data), data_copy)
        os.remove("cptn.pkl")

    def test_Trainer_add_validation_data(self):
        params = init_lstm_params(hidden_size=69)
        trainer = Trainer(params,
                          model='lstm',
                          learning_rate=0.01,
                          batch_size=10,
                          num_epochs=0,
                          sequence_length=16)
        trainer.add_validation_data("")
        self.assertEqual(np.array(trainer.val_data).shape, (704, trainer.sequence_length, 24, 3))

        with open("cptn.pkl", "rb") as f:
            data_copy = pickle.load(f)
        data_copy = make_samples(data_copy, trainer.sequence_length)
        data_copy = np.array(data_copy)

        self.assertEqual(np.array(trainer.val_data).shape, data_copy.shape)
        assert np.array_equal(np.array(trainer.val_data), data_copy)
        os.remove("cptn.pkl")

    def test_Trainer_add_anomaly_data(self):
        params = init_transformer_params(hidden_size=400)
        trainer = Trainer(params,
                          model='transformer',
                          learning_rate=0.0001,
                          batch_size=12,
                          num_epochs=1,
                          sequence_length=100)
        trainer.add_anomaly_data("")
        self.assertEqual(np.array(trainer.anomaly_data).shape, (379, trainer.sequence_length, 24, 3))

        with open("cptn.pkl", "rb") as f:
            data_copy = pickle.load(f)
        data_copy = make_samples(data_copy, trainer.sequence_length)
        data_copy = np.array(data_copy)

        self.assertEqual(np.array(trainer.anomaly_data).shape, data_copy.shape)
        assert np.array_equal(np.array(trainer.anomaly_data), data_copy)
        os.remove("cptn.pkl")

    def test_Trainer_create_validation_set(self):
        params = init_rnn_params(hidden_size=217)
        trainer = Trainer(params,
                          model='rnn',
                          learning_rate=0.0000001,
                          batch_size=9,
                          num_epochs=20,
                          sequence_length=16)
        trainer.add_data("")
        trainer.add_validation_data("")
        self.assertEqual(np.array(trainer.data).shape, (704, trainer.sequence_length, 24, 3))
        self.assertEqual(np.array(trainer.val_data).shape, (704, trainer.sequence_length, 24, 3))
        trainer.create_validation_set(0.5)
        self.assertEqual(np.array(trainer.data).shape, (352, trainer.sequence_length, 24, 3))
        self.assertEqual(np.array(trainer.val_data).shape, (1056, trainer.sequence_length, 24, 3))

        os.remove("cptn.pkl")

    def test_Trainer_set_output_model_name(self):
        params = init_lstm_params(hidden_size=69)
        trainer = Trainer(params,
                          model='lstm',
                          learning_rate=0.01,
                          batch_size=10,
                          num_epochs=0,
                          sequence_length=16)
        trainer.set_output_model_name("test")
        self.assertEqual(trainer.output_model_name, "test")

    def test_Trainer_train_with_rnn(self):
        params = init_rnn_params(hidden_size=264)
        trainer = Trainer(params,
                          model='rnn',
                          learning_rate=0.0000001,
                          batch_size=8,
                          num_epochs=2,
                          sequence_length=8)
        trainer.add_data("")
        trainer.train(save_model=True)

        trainer.create_validation_set(0.2)
        trainer.train(save_model=True)

        trainer.add_anomaly_data("")
        trainer.train(save_model=False)

        trainer.val_data = None

        trainer.train()
        trainer.save_best_model()

        for_delete = load_data("", "pt")
        for i in for_delete:
            os.remove(i)

        for_delete = load_data("", "json")
        for i in for_delete:
            os.remove(i)

        os.remove("cptn.pkl")

