import unittest
import utils as tool


class TestUtils(unittest.TestCase):
    def test_load_data(self):
        self.assertEqual(len(tool.load_data(".", "py")), 4)
        self.assertEqual(len(tool.load_data(".", "mp4")), 1)

    def test_extract_sequential(self):
        sequences = tool.extract_sequential("cptn.mp4", make_mirrors=False)
        sequences_m = tool.extract_sequential("cptn.mp4", make_mirrors=True)
        for seq in sequences:
            self.assertTrue(len(seq) > 1)
            for frame in seq:
                self.assertEqual(frame.shape, (24, 3))

        for seq in sequences_m:
            self.assertTrue(len(seq) > 1)
            for frame in seq:
                self.assertEqual(frame.shape, (24, 3))

    def test_make_samples(self):
        test_seq = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]
        samples = tool.make_samples(test_seq, 3)
        self.assertEqual(len(samples), 7)
        samples = tool.make_samples(test_seq, 4)
        self.assertEqual(len(samples), 5)
        samples = tool.make_samples(test_seq, 5)
        self.assertEqual(len(samples), 3)